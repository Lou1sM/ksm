import os
import numpy as np
import torch
import pickle
import sys
from PIL import Image
from os import listdir
from os.path import join
import struct
from concurrent.futures import ThreadPoolExecutor
from sklearn.datasets import fetch_20newsgroups
from natsort import natsorted
import struct
from datasets import load_dataset, concatenate_datasets
import torchvision.datasets as tdatasets
from tqdm import tqdm
import sys
import os
from scipy.sparse import csr_matrix  # Convert to CSR format if needed
import scipy.io
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



imagenette_synset_dict = {
    'n01440764': 'tench',
    'n02102040': 'spaniel',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
    }

import os
from datasets import load_dataset
import subprocess


def load_pbmc():
    matrix = scipy.io.mmread("filtered_gene_bc_matrices/hg19/matrix.mtx")
    matrix = csr_matrix(matrix)  # Convert to compressed sparse row format
    np_mat = matrix.toarray().T
    return np_mat

def download_and_downsample(example, ignore_cache=False):
    save_dir = "data/msrvtt/raw_vids"
    os.makedirs(save_dir, exist_ok=True)
    vid = example["video_id"]
    url = example["url"]
    out_path = f"{save_dir}/{vid}.mp4"
    if os.path.exists(out_path) and not ignore_cache:
        return out_path

    print(777)
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': f'{save_dir}/{vid}_orig.%(ext)s',
        'quiet': True,
        'cookiesfrombrowser': ('chrome',),
    }

    orig_path = f"{save_dir}/{vid}_orig.mp4"
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        print(888)
        #subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', "-i", orig_path, "-vf", "scale=iw/2:ih/2", "-crf", "32", "-preset", "fast", out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', "-i", orig_path, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "32", "-preset", "fast", out_path], check=True)
        print(999)
        print(os.path.getsize(out_path))
        if os.path.getsize(out_path)==0:
            return orig_path
        else:
            os.remove(orig_path)
            return out_path
    except Exception as e:
        print(f"Failed {vid}: {e}", end=' ')
        if os.path.exists(orig_path):
            print(', returning full vid')
            return orig_path
        else:
            print(', returning None')
            return None

def load_msrvtt(encoder_name, recompute_feats):
    import yt_dlp
    import ffmpeg
    from moviepy.video.io.VideoFileClip import VideoFileClip
    #ds = load_dataset("AlexZigma/msr-vtt")
    ds = load_dataset("friedrichor/MSR-VTT", 'train_9k')
    #all_data = concatenate_datasets([ds["train"], ds["val"]])
    all_data = ds['train']
    vid_save_dir = f"data/msrvtt/{encoder_name}_feats/{encoder_name}_vid_feats"
    text_save_dir = f"data/msrvtt/{encoder_name}_feats/{encoder_name}_text_feats"
    os.makedirs(vid_save_dir, exist_ok=True)
    os.makedirs(text_save_dir, exist_ok=True)
    if encoder_name == 'iv':
        original_sys_path = sys.path.copy()
        sys.path.append(os.path.abspath("/home/louis/amazon_video/InternVideo"))
        from InternVideo2.multi_modality.demo_config import Config, eval_dict_leaf
        from InternVideo2.multi_modality.demo.iv_utils import frames2tensor, setup_internvideo2
        sys.path = original_sys_path
        config = Config.from_file('/home/louis/amazon_video/InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py')
        config = eval_dict_leaf(config)
        config.device = 'cuda'
        config.model.text_encoder.config = '/home/louis/amazon_video/InternVideo/InternVideo2/multi_modality/' + config.model.text_encoder.config
        intern_model, tokenizer = setup_internvideo2(config)
        intern_model.half()
    elif encoder_name=='clip':
        import open_clip
        clip_model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        #clip_model.cuda()
    else:
        sys.exit(f'unrecognised encoder name: {encoder_name}')
    all_feats = []
    all_categories = []
    all_data = natsorted(all_data, key=lambda x: x['video_id'], reverse=True)
    if os.path.exists(unavailable_vids_fp:='data/msrvtt/unavailable-vids-list.txt'):
        with open(unavailable_vids_fp) as f:
            unavailable_vids = f.read().split('\n')
    else:
        unavailable_vids = []
    if os.path.exists(failed_vids_fp:='data/msrvtt/failed-vids-list.txt'):
        with open(failed_vids_fp) as f:
            failed_vids = f.read().split('\n')
    else:
        failed_vids = []
    for i, ex in enumerate(tqdm(all_data)):
        #if i < 1000 or i > 3000:
            #continue
        if ex['video_id'] == 'video8795': # crazy-long video that couldn't get downsampled so gives OOM
            continue
        vid_feats_cached_fp = os.path.join(vid_save_dir, f'{ex["video_id"]}.npy')
        if os.path.exists(vid_feats_cached_fp) and not recompute_feats:
            vid_feats = np.load(vid_feats_cached_fp)
        elif ex['video_id'] in unavailable_vids:
            continue
        elif ex['video_id'] in failed_vids:
            continue
        else:
            cached_fp = download_and_downsample(ex)
            if cached_fp is None:
                unavailable_vids.append(ex['video_id'])
                with open(unavailable_vids_fp, 'w') as f:
                    f.write('\n'.join(unavailable_vids))
                continue
            duration = float(ffmpeg.probe(cached_fp)["format"]["duration"])
            if duration > 1000:
                continue
            try:
                vid = VideoFileClip(cached_fp)
            except OSError as e:
                print(f'Failed reading cached_fp for {ex["video_id"]}: {e}')
                cached_fp = download_and_downsample(ex, ignore_cache=True)
                try:
                    vid = VideoFileClip(cached_fp)
                except OSError as e:
                    print(f'failed with redownload too: {e}')
                    failed_vids.append(ex['video_id'])
                    with open(failed_vids_fp, 'w') as f:
                        f.write('\n'.join(failed_vids))
                    continue

            vid_frames = list(vid.iter_frames())
            n_frames_to_use = 4
            with torch.no_grad():
                if encoder_name == 'iv':
                    batch = frames2tensor(vid_frames, fnum=n_frames_to_use, target_size=(224, 224), device='cuda')
                    vid_feats = intern_model.get_vid_feat(batch)
                else:
                    batch = [vid_frames[round(i*(len(vid_frames)-1)/n_frames_to_use-1)] for i in range(n_frames_to_use)]

                    batch = torch.cat([preprocess(Image.fromarray(x)).unsqueeze(0) for x in batch])
                    batch = batch.squeeze(0).cuda()
                    breakpoint()
                    vid_feats = clip_model.encode_image(batch)
            vid_feats = vid_feats.detach().cpu().numpy()
            vid_feats = vid_feats.mean(axis=0) # mean across frames
            np.save(vid_feats_cached_fp, vid_feats)

        text_feats_cached_fp = os.path.join(text_save_dir, f'{ex["video_id"]}.npy')
        if os.path.exists(text_feats_cached_fp) and not recompute_feats:
            text_feats = np.load(text_feats_cached_fp)
        else:
            with torch.no_grad():
                if encoder_name == 'iv':
                    text_feats = intern_model.get_txt_feat(['\n'.join(ex['caption'])])
                else:
                    tokenized_text = clip_tokenizer(ex['caption']).cuda()
                    text_feats = clip_model.encode_text(tokenized_text)
            text_feats = text_feats.detach().cpu().numpy()
            np.save(text_feats_cached_fp, text_feats)
        feats = (vid_feats + text_feats) / 2
        #feats = text_feats
        assert feats.ndim == 1
        all_feats.append(feats)
        all_categories.append(ex['category'])
    assert len(all_feats) == len(all_categories)
    return np.array(all_feats), np.array(all_categories)

def load_speech_commands():
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    ds = load_dataset('google/speech_commands', 'v0.02')
    labels = np.array(ds['train']['label'] + ds['validation']['label'] + ds['test']['label'])
    n = len(ds['train'])+len(ds['validation'])+len(ds['test'])
    os.makedirs(feat_vecs_dir:='data/speech_commands', exist_ok=True)
    if all(os.path.exists(f'{feat_vecs_dir}/speech_command_featvec{i}.npy') for i in range(n)):
        audio_features = np.stack([np.load(f'{feat_vecs_dir}/speech_command_featvec{i}.npy') for i in range(n)])
    else:
        #raw_audio = np.concatenate([[k['array'] for k in ds[s]['audio'] if k['array'].shape==(16000,)]) for s in ['train', 'test']]) # val files wrong shape for some reason
        raw_audio = [k['array'] for s in ['train', 'validation', 'test'] for k in ds[s]['audio']]
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").cuda()
        bs = 32
        audio_features = []
        for i in tqdm(range(0, len(raw_audio), bs)):
            save_paths = [f'data/speech_commands/speech_command_featvec{i+j}.npy' for j in range(bs)]
            if all(os.path.exists(p) for p in save_paths):
                audio_features += [np.load(fp) for fp in save_paths]
            else:
                batch = [x[:30000] for x in raw_audio[i:i+bs]] # truncate the rare long examples
                inputs = processor(batch, sampling_rate=16000, return_tensors="pt", padding="longest").input_values.cuda()
                batch_feats = model(inputs).last_hidden_state.detach().cpu().mean(axis=1).numpy()
                for sp, f in zip(save_paths, batch_feats):
                    np.save(sp, f)
                    if not f.shape == (768,):
                        breakpoint()
                    audio_features.append(f)
            if any(x.ndim==2 for x in audio_features):
                breakpoint()
        audio_features = np.stack(audio_features)

    return audio_features, labels

def load_ng_feats():
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    text_feats = np.concatenate([np.load(f'data/ng20-featvecs/{fn}') for fn in natsorted(os.listdir('data/ng20-featvecs'))])
    labels = newsgroups.target
    return text_feats, labels

def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    return np.array(img)

def load_im_feats():
    im_feats = []
    class_labels = []
    for class_dir in natsorted(os.listdir('data/im-featvecs')):
        for fn in natsorted(os.listdir(f'data/im-featvecs/{class_dir}')):
            im_feats.append(np.load(f'data/im-featvecs/{class_dir}/{fn}'))
            class_labels.append(class_dir)
    im_feats = np.stack(im_feats)
    class2num = {cl:i for i,cl in enumerate(set(class_labels))}
    labels = np.array([class2num[x] for x in class_labels])
    return im_feats, labels

def load_all_in_tree(im_dir):
    image_paths = [os.path.join(dirpath, fn) for dirpath, dirnames, filenames in os.walk(im_dir) for fn in filenames]
    image_paths = [fp for fp in image_paths if fp.endswith('JPEG')]
    with ThreadPoolExecutor(max_workers=4) as executor:
        loaded_images = list(executor.map(load_image, image_paths))
    return np.stack(loaded_images), image_paths

def load_mnist(split):
    if split == 'both':
        imgs_train, label_train = load_mnist_train_test('train')
        imgs_test, label_test = load_mnist_train_test('test')
        imgs = np.concatenate([imgs_train, imgs_test], axis=0)
        label = np.concatenate([label_train, label_test], axis=0)
    else:
        imgs, label = load_mnist_train_test(split)
    return imgs, label

def load_mnist_train_test(split):
    split = 't10k' if split=='test' else 'train'
    with open(f'data/mnist/{split}-images-idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        imgs = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        imgs = imgs.reshape((size, nrows, ncols))
    with open(f'data/mnist/{split}-labels-idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    return imgs, labels

def load_rand(dset,resize=False):
    if dset=='imagenette':
        dset_dir = 'data/imagenette2/val'
        class_dir = np.random.choice(listdir(dset_dir))
        class_dir_path = join(dset_dir,class_dir)
    elif dset=='dtd':
        class_dir_path = 'data/dtd/suitable'
    fname = np.random.choice(listdir(class_dir_path))
    fpath = join(class_dir_path,fname)
    print(fname)
    return load_fpath(fpath,resize)

def load_usps():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    dtrain=tdatasets.USPS(root='data/usps',train=True,download=True)
    dtest=tdatasets.USPS(root='data/usps',train=False,download=True)
    X = np.concatenate([dtrain.data,dtest.data])
    gtlabels = np.concatenate([dtrain.targets,dtest.targets])
    return X, gtlabels

def load_fpath(fpath,resize,downsample):
    im = Image.open(fpath)
    if resize:
        h,w = im.size[:2]
        aspect_ratio = h/w
        new_h = (224*224*aspect_ratio)**0.5
        new_w = 224*224/new_h
        new_h_int = round(new_h)
        new_w_int = round(new_w)
        max_possible_error = (new_h_int + new_w_int) / 2
        if not (new_h_int*new_w_int - 224*224) < max_possible_error:
            breakpoint()
        if downsample != -1:
            im = im.resize((downsample,downsample))
        im = im.resize((new_h_int,new_w_int))
    im = np.array(im)
    if im.ndim==2:
        im = np.tile(np.expand_dims(im,2),(1,1,3))
    return im

def generate_non_torch_im(dset,resize,subsample):
    if dset=='imagenette':
        dset_dir = 'data/imagenette2/val'
    elif dset=='dtd':
        dset_dir = 'data/dtd/suitable'
    for i in range(subsample):
        if dset=='imagenette':
            num_classes = len(listdir(dset_dir))
            class_dir = listdir(dset_dir)[i%num_classes]
            idx_within_class = i//num_classes
            fname = listdir(join(dset_dir,class_dir))[idx_within_class]
            fpath = join(dset_dir,class_dir,fname)
        elif dset=='dtd':
            try:
                fname = listdir(dset_dir)[i]
            except IndexError:
                print(f"have run out of images, at image number {i}")
                sys.exit()
            fpath = join(dset_dir,fname)
        yield load_fpath(fpath,resize), fpath

def switch_rand_pos(img, switch_pos_x, switch_pos_y):
    if switch_pos_x is None:
        switch_pos_x = np.random.choice(img.shape[0]-1)
    if switch_pos_y is None:
        switch_pos_y = np.random.choice(img.shape[1])
    tmp = img[switch_pos_x+1, switch_pos_y, 0]
    img[switch_pos_x+1, switch_pos_y, 0] = img[switch_pos_x, switch_pos_y, 0]
    img[switch_pos_x, switch_pos_y, 0] = tmp
    assert img.mean()==0.5
    return img

def maybe_cached_dimred(dset_name, raw, dim_red_alg, recompute=False):
    if os.path.exists(dim_red_fp:=f'dim_red_cache/{dim_red_alg}/{dset_name}_{dim_red_alg}ed_X.npy') and not recompute:
        dim_red_X = np.load(dim_red_fp)
    else:
        raw = raw.reshape(raw.shape[0], -1)
        if dim_red_alg=='umap':
            print('loading umap')
            import umap
            if raw.shape[1] > 200:
                print('applying pca')
                raw = PCA(200).fit_transform(raw).astype(np.float32)
            print('applying umap')
            dim_red_X = umap.UMAP(min_dist=0,n_neighbors=10,n_components=2,random_state=42).fit_transform(raw)
        elif dim_red_alg=='tsne':
            dim_red_X = TSNE(2).fit_transform(raw)
        elif dim_red_alg=='pca':
            dim_red_X = PCA(2).fit_transform(raw).astype(np.float32)
        #z = np.unique(dim_red_X.flatten()).astype(float)
        #existing_precision = -np.log(min(np.sort(z)[1:] - np.sort(z)[:-1]))
        #desired_precision = np.exp(-np.log(2)*32)
        #scale_to_max_precision = desired_precision/existing_precision
        #dim_red_X *= scale_to_max_precision
        os.makedirs(os.path.dirname(dim_red_fp), exist_ok=True)
        np.save(dim_red_fp, dim_red_X)
    return dim_red_X

