from typing import Literal, Union
import os
import json
from datasets import load_dataset

from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import FormatTextGenerationSFT, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
import time

class SocialAI_PostsGenerator(TextGeneration):
    system_prompt: str = (
        "You are an AI assistant expert at writing engaging social media posts. "
        "Generate diverse posts that include:\n"
        "- Questions about life decisions\n"
        "- Personal thoughts and reflections\n"
        "- Comments about everyday situations\n"
        "- Opinions about general topics\n"
        "Each post must be natural, engaging, and no more than 200 characters."
    )
    template: str = "Generate a single social media post."
    columns: Union[str, list[str]] = []  # No input columns needed

    def load(self) -> None:
        super().load()

def generate_posts(num_posts=100):
    # Minimal data structure - just need a trigger for each desired generation
    data = [{"prompt": "generate"} for _ in range(num_posts)]
    
    with Pipeline(name="Posts Generator") as pipeline:
        loader = LoadDataFromDicts(data=data, batch_size=10)

        llm = OpenAILLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url="https://inference.api.nscale.com/v1",
            api_key="",
        )

        post_generator = SocialAI_PostsGenerator(
            llm=llm,
            name="post_generator",
            output_mappings={"generation": "post"}
        )

        loader >> post_generator

    # Run the pipeline
    distiset = pipeline.run(use_cache=False)

    print("check the distiset", distiset)
    
    # Extract generated posts
    generated_posts = []
    for item in distiset['default']['train']:
        if 'post' in item and item['post'] is not None:
            generated_posts.append({"post": item['post']})

    # Save to JSON
    with open('generated_posts.json', 'w') as f:
        json.dump(generated_posts, f, indent=2)

    print(f"Generated {len(generated_posts)} posts and saved to generated_posts.json")
    return generated_posts

if __name__ == "__main__":
    start_time = time.time()
    posts = generate_posts(num_posts=100)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    # Print some examples
    print("\nExample generated posts:")
    for i, post in enumerate(posts[:10]):
        print(f"{i+1}. {post['post']}")
