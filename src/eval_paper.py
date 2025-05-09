from tqdm import tqdm
import torch
from pipeline_flux import FluxPipeline
from transformer_flux import FluxTransformer2DModel
import os
import random
from safetensors.torch import load_file
from patch_conv import convert_model

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
    
    "a cyberpunk cat with a neon sign that says \"WALD\"",
    
    "A very detailed and realistic full body photo set of a tall, slim, and athletic Shiba Inu in a white oversized straight t-shirt, white shorts, and short white shoes.",
    
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    
    "a stunning and luxurious bedroom carved into a rocky mountainside seamlessly blending nature with modern design with a plush earth-toned bed textured stone walls circular fireplace massive uniquely shaped window framing snow-capped mountains dense forests"
]

prompts_naples = [
            "A breathtaking view of the Gulf of Naples at sunset, with Mount Vesuvius in the background and the city lights reflecting on the water.",
            "A narrow and picturesque alley in the historic center of Naples, with laundry hanging out to dry between flower-filled balconies and the inviting aroma of freshly baked pizza in the air.",
            "A steaming Margherita pizza, with Buffalo Mozzarella DOP from Campania, San Marzano tomatoes, and fresh basil, on a table in a typical Neapolitan trattoria.",
            "A hyperrealistic image of the Veiled Christ in the Sansevero Chapel, with its marble sculpture appearing to breathe beneath the transparent veil.",
            "A festive crowd in Piazza del Plebiscito during the Feast of San Gennaro, with the procession of the saint's bust and the anticipation of the miracle of the liquefaction of the blood.",
            "An explosion of colors and scents at the Porta Nolana market, with stalls overflowing with fruits, vegetables, fresh fish, and exotic spices.",
            "An abstract image inspired by the shapes and colors of the Naples Metro, with its art stations and mosaics that tell the city's history.",
            "A stylized portrait of Pulcinella, the quintessential Neapolitan mask, with his white hat, black mask, and a mischievous and carefree air.",
            "An aerial view of Castel dell'Ovo, with its distinctive shape and its striking location on the islet of Megaride.",
            "An artistic interpretation of a 'sfogliatella riccia', with its shell-like shape, crispy layered pastry, and creamy filling of ricotta cheese and candied fruit.",
            "A hyperrealistic image of the SSC Napoli logo, with its distinctive 'N' and blue color, surrounded by a dynamic, swirling vortex of blue smoke and light, evoking the team's energy and passion.",
            "A wide, panoramic view of the Stadio Diego Armando Maradona, formerly San Paolo, during a night match, filled with cheering fans, flares, and banners in the team's colors.",
            "A stylized portrait of Diego Armando Maradona in his prime, wearing the SSC Napoli jersey, with a halo of golden light symbolizing his legendary status in the city.",
            "A close-up, dramatic shot of a group of SSC Napoli ultras, their faces painted in the team's colors, holding up a massive banner with a powerful message of support, in a dimly lit, atmospheric setting.",
            "An abstract image representing the sound of the SSC Napoli anthem being sung by thousands of fans, visualized as vibrant, swirling patterns of blue and white, with dynamic lines and shapes.",
            "A detailed, realistic depiction of the SSC Napoli jersey, with its iconic blue color and sponsor logos, hanging in a locker room, with a sense of anticipation and excitement before a big match.",
            "A conceptual image showing the city of Naples rising up to cheer on SSC Napoli, with buildings and streets morphing into cheering fans, all bathed in a blue light.",
            "A fantastical scene depicting the SSC Napoli mascot, the donkey, as a powerful, mythical creature, leading a charge of blue and white energy across a surreal, dreamlike landscape.",
            "A macro shot of confetti in the SSC Napoli colors, blue and white, falling from the sky after a victory, creating a sense of celebration and joy.",
            "An artistic representation of the bond between SSC Napoli and the city, with the team's colors and symbols interwoven into the fabric of the city's architecture and culture, creating a vibrant, dynamic tapestry.",
            "A jubilant scene of SSC Napoli players lifting the Scudetto trophy, surrounded by fireworks and cheering fans at the Stadio Diego Armando Maradona.",
            "A vibrant street in Naples, overflowing with celebrating fans, blue and white flags, and street art commemorating the Scudetto victory.",
            "A close-up shot of a Napoli fan's face, filled with joy and tears, celebrating the Scudetto win.",
            "An aerial view of Naples, with the city lit up in blue and white, showcasing the widespread celebrations after the Scudetto victory.",
            "A powerful image of the city of Naples embracing the SSC Napoli team, symbolizing the deep connection between the club and its city after winning the Scudetto.",
           
        ]

prompt_gemini = [ "A dreamlike landscape generated with intricate details: bioluminescent trees with leaves emitting a soft, pulsating light, reflected in a crystalline pond populated by translucent aquatic creatures. The scene is enveloped in an ethereal mist and illuminated by a moon with shifting hues.",
            "Hyperrealistic representation of a futuristic organic sculpture, fused with finely detailed mechanical elements. The surface exhibits complex textures, with iridescent metallic reflections that capture a dynamic light source from an unseen origin. The background is a smooth gradient of complementary colors.",
            "A stylized portrait of a mystical entity, with an aura of vibrant energy emanating from its body. The facial details are enigmatic and profound, with eyes that glow with an intense inner light. Ethereal ornaments float around the figure, creating a sense of movement and magic.",
            "Visualization of a lush alien ecosystem, with plants of unexpected geometric shapes and saturated colors. Light filters through a dense canopy of exotic foliage, creating dramatic plays of shadows and light. Iridescent particles float in the air.",
            "An abstract macro photograph of colliding water droplets, capturing the precise moment of their fusion into sculptural liquid forms. The water's surface reflects a rainbow of colors, with microscopic details revealing surface tension and the dynamics of motion.",
            "Conceptual illustration of a utopian flying city, with sinuous and organic architectures suspended among the clouds. Beams of golden light illuminate the structures, connecting them through ethereal bridges and lush hanging gardens. Futuristic vehicles traverse the skies with luminous trails.",
            "Creation of a complex and hypnotic geometric pattern, with three-dimensional shapes that intersect and overlap harmoniously. The color palette is vibrant and contrasting, with subtle gradients that add depth and dynamism to the composition.",
            "An artistic interpretation of a soundscape, where sound waves visually manifest as luminous and colorful ripples propagating through an ethereal space. Higher frequencies are represented by sharp and brilliant forms, while lower frequencies by soft and enveloping curves.",
            "Visualization of a sentient artificial intelligence in the form of an ethereal and luminous figure, surrounded by abstract data streams that intertwine like filaments of light. The figure's eyes express awareness and intellectual depth.",
            "Representation of a shimmering and unstable dimensional portal, opening onto a surreal and unknown landscape. The energy emanating from the portal distorts the surrounding space, creating unexpected optical effects and light refractions."]

list_of_checkpoints_flux_models_2k = [
    # "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_VAE_SE_WAV_ATT_AESTHETIC_2K/checkpoint-2000", #HELP non funziona
    # "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_VAE_SE_WAV_ATT_AESTHETIC_2048/checkpoint-2000", #help img non belle
    
    "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_original_trained_by_me/checkpoint-2000",
    "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_VAE_SE_WAV_ATT_LAION/checkpoint-2000",
    "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/URAE_VAE_SE_WAV_ATT_LAION_2048/checkpoint-2000"
]

list_of_checkpoints_flux_models_4k = [  
    "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors",
    # "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/4K_URAE_VAE_SE_WAV_ATT_AESTHETIC_4096/checkpoint-2000/adapter_weights.safetensors",
#    "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/4K_urae_original/ckpt_xxx/urae_4k_adapter.safetensors", 
    ]

# Create a mapping of prompt lists to their names
prompt_list_names = {
    "prompts_URAE": prompts_URAE,
    "prompts_diff4k": prompts_diff4k,
    "prompts_comp_diff4k": prompts_comp_diff4k,
    "prompts_naples": prompts_naples,
    "prompts_gemini": prompt_gemini,
}

def get_pipe_2k(cache_dir = "/leonardo_scratch/large/userexternal/lsigillo", device_str = "cuda:0"):
    #load original model
    bfl_repo="black-forest-labs/FLUX.1-dev"
    device = torch.device(device_str)
    dtype = torch.bfloat16
    transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, cache_dir=cache_dir)
    pipe = FluxPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=dtype, cache_dir=cache_dir)
    pipe.scheduler.config.use_dynamic_shifting = False
    pipe.scheduler.config.time_shift = 10
    return pipe.to(device)

from eval_4k import load_model_4k
def get_pipe_4k(cache_dir = "/leonardo_scratch/large/userexternal/lsigillo",
                device_str = "cuda:0",
                ckpt_path_4k = "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/4K_URAE_VAE_SE_WAV_ATT_LAION_4096/checkpoint-2000/adapter_weights.safetensors",
                ):
    return load_model_4k(
        device_str, 
        cache_dir, 
        ckpt_lora_weights_2k = "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/urae_2k_adapter.safetensors",
        ckpt_path_4k = ckpt_path_4k,
        cache_path_adapter='/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/ckpt/_urae_4k_adapter_dev.safetensors',
    )

def generate_images(pipe, list_of_checkpoints_flux_models, 
                    current_prompts, height_options, width_options, gen_seed, 
                    output_dir, same_hxw):

    for checkpoint_path in tqdm(list_of_checkpoints_flux_models):
        #our trained URAE
        if "4K" in checkpoint_path:
            state_dict = load_file(checkpoint_path, "cpu")
            m, u = pipe.transformer.load_state_dict(state_dict, strict=False)
            assert len(u) == 0
            pipe.vae = convert_model(pipe.vae, splits=4)
            name_exp = checkpoint_path.split("/")[-3]

        else:
            pipe.load_lora_weights(checkpoint_path)
            # pipe = pipe.to(device)
            name_exp = checkpoint_path.split("/")[-2]
        
        # Get the name of the list from the mapping
        curr_list = prompt_list_names[current_prompts]
        os.makedirs(os.path.join(output_dir, name_exp, f"paper_{current_prompts}"), exist_ok=True)

        for idx, prompt in tqdm(enumerate(curr_list)):
            if same_hxw:
                height = max(height_options)
                width = max(width_options)
            else:
                while True:
                    height = random.choice(height_options)
                    width = random.choice(width_options)
                    if height != min(height_options) or width != min(width_options):
                        break
            image = pipe(
                prompt,
                height=height,
                width=width,
                guidance_scale=3.5,
                num_inference_steps=28,
                max_sequence_length=512,
                generator=gen_seed,
                ntk_factor=10,
                num_images_per_prompt=1,
                proportional_attention=True
            ).images
            for i, img in enumerate(image):
                img.save(os.path.join(output_dir, name_exp, f"paper_{current_prompts}", f"{idx:05d}_{height}x{width}_{i}.jpg"))
            del image
            torch.cuda.empty_cache()


def main(current_prompts="prompts_comp_diff4k", k2=True, same_hxw=False):
    
    # Use the current prompts list you're working with
    # current_prompts = "prompts_comp_diff4k"

    device_str = "cuda:0"
    height_options = [1560, 2048]
    width_options = [1560, 2048]
    height_options_4k = [2160, 4096]
    width_options_4k = [2160, 4096]
    seed= 42
    output_dir = "/leonardo_scratch/fast/IscrC_UniMod/luigi/HighResolutionWav/src/output"
    gen_seed = torch.manual_seed(seed=seed)
    
    if k2:
        generate_images(
            get_pipe_2k(
                device_str=device_str
            ), 
            list_of_checkpoints_flux_models_2k, 
            current_prompts, 
            height_options, 
            width_options, 
            gen_seed, 
            output_dir,
            same_hxw
        )
    else:
        generate_images(
            get_pipe_4k(
                device_str=device_str, 
            ), 
            list_of_checkpoints_flux_models_4k, 
            current_prompts, 
            height_options_4k, 
            width_options_4k, 
            gen_seed, 
            output_dir,
            same_hxw
        )




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate images with specified prompts.")
    parser.add_argument("--current_prompts", type=str, default="prompts_comp_diff4k", help="Prompt list to use.")
    parser.add_argument("--k2", action="store_true", help="Use 2k model.")
    parser.add_argument("--same_hxw", action="store_true", help="use same h x w in generation")

    args = parser.parse_args()
    main(current_prompts=args.current_prompts, k2=args.k2, same_hxw=args.same_hxw)