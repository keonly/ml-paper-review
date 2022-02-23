import copy
import clip
import pickle
import numpy as np
import argparse
from PIL import Image

import torch
from torch.nn import functional as F

from stylegan2.models import Generator
from singleChannel import *




imagenet_templates = [
    'a bad photo of a {}.',
#    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]



def GetDt(classnames, model):
    """
    classnames: [target, neutral]
    model: CLIP 
    """
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in imagenet_templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        text_features= torch.stack(zeroshot_weights, dim=1).cuda().t()
    
    dt=text_features[0]-text_features[1] # target embedding - neutral embedding
    dt=dt.cpu().numpy()
    dt=dt/np.linalg.norm(dt)
    return dt

def GetBoundary(fs3, dt, threshold, style_space=None, style_names=None):
    """
    Measure cosine similarity of StyleSpace channels and Text representation
    Filter out similarity lower thatn the threshold
    Split (6048, ) into corresponding scale using SplitS (to be passed to StyleGAN G)
    
    fs3: collection of predefined styles (6048, 512)
    dt: Embedding delta text (, 512)
    return:
        ds_imp: deviation of style
            channelwise style movement * dText
        num_c: number of channels manipulated (number of nonzero elements)
    """
    #! TODO
    pass
    
    if style_space is None:
        return ds_imp, num_c
    else:
        boundary_tmp2, dlatents=SplitS(ds_imp, style_names, style_space)
        return boundary_tmp2, num_c, dlatents
            
def SplitS(ds_p, style_names, style_space):
    
    all_ds=[]
    start=0
    tmp="./npy/ffhq/"+'S'
    with open(tmp, "rb") as fp:   #Pickling
        _, dlatents=pickle.load(fp)
    tmp="./npy/ffhq/"+'S_mean_std'
    with open(tmp, "rb") as fp:   #Pickling
        m, std=pickle.load( fp)

    for i, name in enumerate(style_names):

        if "torgb" not in name:
            tmp=style_space[i].shape[1]
            end=start+tmp
            tmp=ds_p[start:end] * std[i]
            all_ds.append(tmp)
            start=end
        else:
            tmp = np.zeros(len(dlatents[i][0]))
            all_ds.append(tmp)
    return all_ds, dlatents

def MSCode(dlatent_tmp, boundary_tmp, manipulate_layers, num_images, alpha, device):
    """
    Returns manipulated Style Space
    """
    step=len(alpha)
    dlatent_tmp1=[tmp.reshape((num_images,-1)) for tmp in dlatent_tmp]
    dlatent_tmp2=[np.tile(tmp[:,None],(1,step,1)) for tmp in dlatent_tmp1]

    l=np.array(alpha)
    l=l.reshape([step if axis == 1 else 1 for axis in range(dlatent_tmp2[0].ndim)])
    if manipulate_layers is None:
        tmp=np.arange(len(boundary_tmp))
    else:
        tmp = [manipulate_layers]

    for i in tmp:
        dlatent_tmp2[i] += l * boundary_tmp[i]
    
    codes=[]
    for i in range(len(dlatent_tmp2)):
        tmp=list(dlatent_tmp[i].shape)
        tmp.insert(1,step)
        code = torch.Tensor(dlatent_tmp2[i].reshape(tmp))
        codes.append(code.to(device))
    return codes

def conv_warper(layer, input, style, noise):
    # the conv should change
    conv = layer.conv
    batch, in_channel, height, width = input.shape
 
    # style = self.modulation(latent).view(batch, 1, in_channel, 1, 1) : This part is removed to inject given style
    style = style.view(batch, 1, in_channel, 1, 1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample: # up==2
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
    # image + self.weight * noise
    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    
    return out

def decoder(G, style_space, latent, noise):
    """
    G: stylegan generator
    style_space: S space of size 9088 (6048 for conv / 3040 for torgb)
    latent: W space vector of size 512
    noise: noise for each layer of styles (predefined from encoder step)
    """
    out = G.input(latent)
    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip = G.to_rgb1(out, latent[:, 0])
    i = 2; j = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip = to_rgb(out, latent[:, j + 2], skip) # style space manipulation not used in torgb
        i += 3; j += 2
    image = skip

    return image

def encoder(G, latent): 
    noise_constants = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    style_space = []
    style_names = []
    # rgb_style_space = []
    style_space.append(G.conv1.conv.modulation(latent[:, 0]))
    res=4
    style_names.append(f"b{res}/conv1")
    style_space.append(G.to_rgbs[0].conv.modulation(latent[:, 0]))
    style_names.append(f"b{res}/torgb")
    i = 1;j=3

    for conv1, conv2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], G.to_rgbs
    ):
        res=2**j
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_names.append(f"b{res}/conv1")
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        style_names.append(f"b{res}/conv2")
        style_space.append(to_rgb.conv.modulation(latent[:, i + 2]))
        style_names.append(f"b{res}/torgb")
        i += 2; j += 1
        
    return style_space, style_names, noise_constants

def visual(output, save=False, name="original"):
    output = (output + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    output = output[0].detach().cpu().permute(1,2,0).numpy()
    output = (output*255).astype(np.uint8)
    img = Image.fromarray(output)
    if save:
        img.save(f"results/{name}.png")
    return output

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Configuration for styleCLIP')
    parser.add_argument('--neutral', type=str, default='A girl', help='Neutral text without additional information of the image')
    parser.add_argument('--target', type=str, default="A girl is rich", help='Target text to manipulate the image generated')
    parser.add_argument('--alpha', type=float, default=10.0, help='Manipulation strength, Between -10 ~ 10')
    parser.add_argument('--beta', type=float, default=0.08, help='Manipulation threshold, Between 0.08 ~ 3')
    parser.add_argument('--top', type=int, default=5, help='Number of layers to be manipulated')
    parser.add_argument('--use_w', action="store_true", help='Use W plus space to manipulate otherwise manipulate in S space')
    parser.add_argument('--file_path', type=str, default="./npy/ffhq/", help="Path where W/S statistcs are stored")
    parser.add_argument('--pca_file', type=str, default="stylegan2-ffhq_style_c50_n1000000_w.npz", help="File name of pca components")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Torch device on {device}")
    config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}

    #* Load StyleGAN2 pretrained generator
    generator = Generator(
            size= 1024,
            latent_dim=config["latent"],
            n_mlp=config["n_mlp"],
            channel_multiplier=config["channel_multiplier"]
        )

    generator.load_state_dict(torch.load("model/stylegan2-ffhq-config-f.pt")['g_ema'])
    generator.eval()
    generator.to(device)

    #* Load W space latent vector of size 1, 18, 512
    latent = torch.load("./model/latent.pt")

    #* Load style space of S from the latent/ Extract noise constant from pretrained model
    style_space, style_names, noise_constants = encoder(generator, latent)
    image = decoder(generator, style_space, latent, noise_constants)
    tmp = visual(image, save=True, name="original")

    model, _ = clip.load("ViT-B/32", device=device)
    
    np.set_printoptions(suppress=True)
    classnames = [args.target, args.neutral]

    dt = GetDt(classnames, model)
    #* Load PC components 
    dump_name = args.file_path + args.pca_file
    n_comp = [int(i.strip("c")) for i in args.pca_file.split("_") if "c" in i][0]
    print(f"Number of PC: {n_comp}")
    comp = load_components(dump_name) # 1, n_comp, 512

    #* Load W plus latents inverted from FFHQ dataset
    tmp = args.file_path+"W.npy" 
    w_samples = np.load(tmp) # 100000, 512
    w_std = w_stat(w_samples, std=True) # 1, 512 -> Used as manipulation strength
    
    #* Manipulation in W Plus space
    if args.use_w:
        fs3 = np.load('./npy/ffhq/W_manip_50comps_1000imgs.npy') # 18*n_comp, 512
        ds, num_c = GetBoundary(fs3, dt, args.beta)
        ds = ds.reshape([18, n_comp])
        candidates = np.transpose(np.nonzero(ds)) # Manipulation candidates: [array of layer indices, array of component indices] 
        ds_nonzero = ds[np.nonzero(ds)] # Text & we channel similarity values of candidates
        
        manipulations = []
        sim = []
        max_cand = candidates[0]
        max = ds_nonzero[0]
        for candidate, val in zip(candidates[1:], ds_nonzero[1:]):
            print(candidate,"\t", val)
            if max_cand[0] == candidate[0]:
                if np.abs(max) < np.abs(val):
                    max = val
                    max_cand = candidate
            else: 
                # Append the largest candidate for previous layer
                manipulations.append(max_cand)
                sim.append(max)
                # Moving on to the next layer
                max =  val
                max_cand = candidate
                
        manipulations = np.array(manipulations)
        
        print(f"number of layers manipulated: {num_c}")

        layers, components = manipulations.T
        print(f"Dot product similarity scores: {sim}")
        print(f"Layers: {layers}")
        print(f"Components: {components}")
        layers = [x for _, x in sorted(zip([abs(s) for s in sim], layers), reverse=True)][:args.top]
        components = [x for _, x in sorted(zip([abs(s) for s in sim], components), reverse=True)][:args.top]
        print(f"layers: {layers} components: {components}")
        layer_iter = iter(layers)
        components_iter = iter(components)
        for i in range(18):
            if i in layers:
                l = next(layer_iter)
                c = next(components_iter)
                print(f"layer: {l}, compoenent {c}")
                latent[:, i, :] += torch.Tensor(np.abs(ds[l, c])*w_std*args.alpha*comp[c]).squeeze().to(device) # 
        style_space, style_names, noise_constants = encoder(generator, latent)
        image = decoder(generator, style_space, latent, noise_constants)
        _ = visual(image, save=True, name=args.target)

    else:
        fs3 = np.load('./npy/ffhq/fs3.npy') # 6048, 512
        boundary_tmp2, c, dlatents = GetBoundary(fs3, dt, args.beta, style_space, style_names) # Move each channel by dStyle
        dlatents_loaded = [s.cpu().detach().numpy() for s in style_space]
        manipulated_s= MSCode(dlatents_loaded, boundary_tmp2, manipulate_layers=None, num_images=1, alpha=[args.alpha], device=device)
        image = decoder(generator, manipulated_s, latent, noise_constants)
        tmp = visual(image, orig=False)
        print(f"Generated Image {args.target}")
        
