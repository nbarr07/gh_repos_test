from datasets import load_dataset

from distilabel.models.image_generation import OpenAIImageGeneration
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns
from distilabel.steps.tasks import ImageGeneration

class CustomOpenAIImageGeneration(OpenAIImageGeneration):
    use_offline_batch_generation: bool = False

ds = load_dataset("dvilasuero/finepersonas-v0.1-tiny", split="train").select(range(3))

with Pipeline(name="image_generation_pipeline") as pipeline:
    igm = CustomOpenAIImageGeneration(
        model="black-forest-labs/FLUX.1-schnell",
        api_key="",
        base_url="https://inference.api.nscale.com/v1",
    )
    print("checkpoint 1")
    igm.load() 
    print("checkpoint 2")
    img_generation = ImageGeneration(
        name="dalle_generation",
        image_generation_model=igm,
        input_mappings={"prompt": "persona"},
        generation_kwargs={
            "size": "1024x1024",
            "quality": "standard",
            "style": "natural",
        }
    )
    print("checkpoint 3")
    keep_columns = KeepColumns(columns=["persona", "model_name", "image"])
    print("checkpoint 4")
    img_generation >> keep_columns


if __name__ == "__main__":
    print("checkpoint 5")
    distiset = pipeline.run(use_cache=False, dataset=ds)
    print("checkpoint 6")
    if len(distiset) > 0:  # Only process if we have data
        distiset = distiset.transform_columns_to_image("image")
        hf_dataset = distiset.to_dataset()
        hf_dataset.save_to_disk("output_dataset")
    else:
        print("No data was generated!")
    print(distiset)
    # Save the images as `PIL.Image.Image`
    distiset = distiset.transform_columns_to_image("image")
    # save the dataset to a csv
    new_df = distiset.to_pandas()
    new_df.to_csv("distiset.csv")
    print("checkpoint 7")
    #distiset.push_to_hub("plaguss/test-finepersonas-v0.1-tiny-flux-schnell")
