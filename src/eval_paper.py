from tqdm import tqdm
import torch
from pipeline_flux import FluxPipeline
from transformer_flux import FluxTransformer2DModel
import os

prompts_URAE = [
    "girl with pink hair, vaporwave style, retro aesthetic, cyberpunk, vibrant, neon colors, vintage 80s and 90s style, highly detailed.",
    
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
    
    "A whimsical village scene is nestled within an enormous teacup, where winding cobblestone streets and quaint cottages create a surreal microcosm reminiscent of dreamlike landscapes. From a high vantage point, the viewer looks down on the diminutive inhabitants going about their day-to-day activities, contrasting the juxtaposition of innocence and chaos in this fantastical setting. Soft, ethereal lighting envelops the scene, creating an atmosphere of tranquility as ordinary life intertwines with elements of fantasy and absurdity to weave a captivating visual narrative that invites the audience into an extraordinary world where the commonplace becomes surreal.",
    
    "A surreal landscape depicting an ethereal fusion of natural beauty and fantastical architecture, reminiscent of Salvador Dali's dreamlike paintings. From above the clouds, one gazes upon a colossal tower emerging from the earth, its intricate gears visible as it merges seamlessly with a tranquil mountain lake. The scene is bathed in an otherworldly glow, casting lavender and gold hues across the sky, while delicate cherry blossoms flutter gently in the foreground, adding a sense of serenity to this breathtaking vision where time and nature intertwine.",
    
    "Imagine an enchanting scene where the dreamy allure of Klimt's celestial brushstrokes intertwines with the magical wonder of Studio Ghibli animations, showcasing a mesmerizing underwater realm teeming with life. In this captivating portrait, a luminous mermaid princess emerges from a vibrant coral landscape, her iridescent silver hair flowing like cascading waterfalls as she sings a hauntingly beautiful melody that resonates through the depths. From an aerial perspective, the scene unfolds in a surreal and awe-inspiring manner, capturing both the ethereal grace of this mythical creature and the breathtaking beauty of her aquatic surroundings, creating a harmonious blend of fantasy and reality that invites viewers to lose themselves in its enchanting embrace.",
    
    "A whimsical portrait of an otherworldly fairy with luminescent wings captures her enchanting features from an unconventional angle beneath a floating moonlit garden. Rendered in high-definition digital art, her luminous eyes gleam like stars and her flowing hair seems caught in an unseen breeze, while the intricate details of shimmering scales on her wings and petals entwined within her tresses contribute to the overall sense of wonder. The soft focus of distant flowers and glistening dewdrops engulfs the scene in a dreamy atmosphere, harmoniously blending with the fairy's ethereal presence in this captivating tableau that evokes feelings of enchantment and wonder.",
    
    "Steampunk airship floating above a misty Victorian cityscape, intricate brass and copper mechanical details, golden hour lighting, billowing clouds, detailed architectural elements, rich warm color palette, cinematic composition.",
    
    "Create an image in the surrealistic style, capturing a unique bird's-eye view of a whimsical scene where a tall giraffe stands atop an oversized piano in a lush garden filled with oversized, vibrant flowers. The giraffe appears to play an invisible melody on the keys, its long neck bending gracefully over the keyboard while its feet dance rhythmically on the pedals. This striking juxtaposition should evoke a sense of wonder and enchantment, blending elements of natural beauty with unexpected musical elements in a dreamlike atmosphere.",
    
    "A sleek black luxury sedan parked on a rain-soaked city street at night, reflecting neon lights from nearby buildings. The wet pavement glistens, and the car's smooth curves are highlighted by the ambient glow of the urban environment.",
    
    "Barbarian woman riding a red dragon, holding a broadsword, in gold armour.",
    
    "A person wearing a Spider-Man suit in the game Half-Life Alyx.",
    
    "A person staring into a lucid dream world with an adventure waiting.",
    
    "A dreamlike landscape depicting an ethereal giraffe with elongated limbs and neck gracefully floating above a surreal desert oasis. This captivating scene is captured from a low-angle perspective, evoking a sense of wonder as the giraffe appears to defy gravity in a whimsical juxtaposition of elements. The image combines vibrant colors and melting textures to create an imaginative and thought-provoking vision that blurs the lines between reality and fantasy, inviting viewers to explore the depths of their own imagination."

    "Craft an image in the surreal digital art style, depicting a dreamlike portrait of a young woman whose face merges with a complex floral arrangement. This scene is viewed from an elevated perspective, giving it a unique 'bug's-eye view', making her appear to be enveloped within a grand garden teeming with vivid, distorted blooms emerging directly from her body and hair. The composition should inspire awe and intrigue as the petals weave into her facial contours, creating a fantastical union of nature and human form. A gentle, diffused lighting bathes this captivating scene, casting delicate shadows that dance across the textures of both the flowers and her translucent skin.",

    "An enigmatic dream landscape is captured in this whimsical scene, where an intricate tower with distorted, fluid architecture dominates the horizon. The sky above is filled with floating islands, their shapes shifting as if viewed through a fisheye lens, emphasizing the surreal nature of the setting. In the foreground, a curious child stands on one of these drifting lands, wide-eyed and flowing-haired, gazing up at the enigmatic tower with a mix of awe and unease, evoking an emotional spectrum that ranges from curiosity to uneasiness in this fantastical landscape."
]

prompts_diff4k = [

    "A vast colony of king penguins densely populates a rocky shore, with numerous individuals standing closely together against a backdrop of ocean waves.",
    
    "A picturesque riverside scene featuring a medieval castle atop a hill, surrounded by vibrant autumn foliage, with colorful homes lining the waterfront and several boats docked along the river.",
    
    "A sea turtle glides gracefully through crystal-clear turquoise water above a school of small fish, with sunlight reflecting off the surface.",
    
    "A lone astronaut floats in space, gazing at a swirling black hole surrounded by vibrant landscapes, rivers, and clouds below.",
    
    "A close-up of a fox's face, partially covered in snow, with sharp ears and bright, alert eyes. The snowy backdrop adds a serene atmosphere to the scene.",
    
    "A majestic polar bear sits near a rocky shore, reflecting in calm waters under a full moon amid a dramatic, cloudy sky filled with flying birds. A small figure stands in the distance, observing the scene.",
    
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
    
    "A gorgeously rendered papercraft world of a coral reef, rife with colorful fish and sea creatures.",
    
    "A litter of golden retriever puppies playing in the snow. Their heads pop out of the snow, covered in.",
    
    "A young man at his 20s is sitting on a piece of cloud in the sky, reading a book.",
    
    "A petri dish with a bamboo forest growing within it that has tiny red pandas running around.",
    
    "Extreme close up of a 24 year old woman's eye blinking, standing in Marrakech during magic hour, cinematic film shot in 70mm, depth of field, vivid colors, cinematic.",
    
    "3D animation of a small, round, fluffy creature with big, expressive eyes explores a vibrant, enchanted forest. The creature, a whimsical blend of a rabbit and a squirrel, has soft blue fur and a bushy, striped tail. It hops along a sparkling stream, its eyes wide with wonder. The forest is alive with magical elements: flowers that glow and change colors, trees with leaves in shades of purple and silver, and small floating lights that resemble fireflies. The creature stops to interact playfully with a group of tiny, fairy-like beings dancing around a mushroom ring. The creature looks up in awe at a large, glowing tree that seems to be the heart of the forest."
]

prompts_comp_diff4k = [
    "A litter of golden retriever puppies playing in the snow. Their heads pop out of the snow, covered in.",
    
    "Eiffel Tower was Made up of more than 2 million translucent straws to look like a cloud, with the bell tower at the top of the building, Michel installed huge foam-making machines in the forest to blow huge amounts of unpredictable wet clouds in the building's classic architecture.",
    
    "Close-up photos of models, hazy light and shadow, laser metal hair accessories, soft and beautiful, light gold pupils, white eyelashes, low saturation, real skin details, clear pores and fine lines, light reflection and refraction, ultra-clear, cinematography, award-winning works.",
    
    "A curvy timber house near a sea, designed by Zaha Hadid, represent the image of a cold, modern architecture, at night, white lighting, highly detailed.",
    
    "a cyberpunk cat with a neon sign that says \"Fast\"",
    
    "A very detailed and realistic full body photo set of a tall, slim, and athletic Shiba Inu in a white oversized straight t-shirt, white shorts, and short white shoes.",
    
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    
    "a stunning and luxurious bedroom carved into a rocky mountainside seamlessly blending nature with modern design with a plush earth-toned bed textured stone walls circular fireplace massive uniquely shaped window framing snow-capped mountains dense forests"
]

list_of_checkpoints_flux_models = [
    # "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/4K_URAE_VAE_SE_WAV_ATT_AESTHETIC_4096/checkpoint-1000",
    # "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-1000",
    
    
    # "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_VAE_SE_WAV_ATT_AESTHETIC_2K/checkpoint-2000", #HELP non funziona
    "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_VAE_SE_WAV_ATT_AESTHETIC_2048/checkpoint-2000", #help img non belle
    
    "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_original_trained_by_me/checkpoint-2000",
    "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_VAE_SE_WAV_ATT_LAION/checkpoint-2000",
    "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_VAE_SE_WAV_ATT_LAION_2048/checkpoint-2000"
]


# Create a mapping of prompt lists to their names
prompt_list_names = {
    "prompts_URAE": prompts_URAE,
    "prompts_diff4k": prompts_diff4k,
    "prompts_comp_diff4k": prompts_comp_diff4k
}
# Use the current prompts list you're working with
current_prompts = "prompts_diff4k"

device_str = "cuda:0"
height = 2048
width = 2048
seed= 42
output_dir = "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/output"
gen_seed = torch.manual_seed(seed=seed)

#load original model
bfl_repo="black-forest-labs/FLUX.1-dev"
cache_dir = "/leonardo_scratch/large/userexternal/lsigillo"
device = torch.device(device_str)
dtype = torch.bfloat16
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, cache_dir=cache_dir)
pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype, cache_dir=cache_dir)
pipe.scheduler.config.use_dynamic_shifting = False
pipe.scheduler.config.time_shift = 10

for checkpoint_path in tqdm(list_of_checkpoints_flux_models):
    #our trained URAE
    pipe.load_lora_weights(checkpoint_path)
    pipe = pipe.to(device)
    name_exp = checkpoint_path.split("/")[-2]
    
    # Get the name of the list from the mapping
    curr_list = prompt_list_names[current_prompts]
    os.makedirs(os.path.join(output_dir, name_exp, f"paper_{current_prompts}"), exist_ok=True)

    for idx, prompt in tqdm(enumerate(curr_list)):
        # image = TextToImageModel(prompt)
        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=3.5,
            num_inference_steps=28,
            max_sequence_length=512,
            generator=gen_seed,
            ntk_factor=10,
            proportional_attention=True
        ).images[0]
        # TextToImageModel is the model you want to evaluate
        image.save(os.path.join(output_dir, name_exp, f"paper_{current_prompts}", f"{idx:05d}.jpg"))