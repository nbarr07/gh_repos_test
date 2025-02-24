import argparse
import asyncio
import aiohttp
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class LLMStressTester:
    def __init__(self, endpoint_url, api_key=None, prompt_templates=None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.prompt_templates = prompt_templates or [
            "Explain the concept of {topic} in simple terms.",
            "Write a short paragraph about {topic}.",
            "What are the key components of {topic}?",
            "Compare and contrast {topic} with another related concept."
        ]
        self.topics = [
            "machine learning", "climate change", "quantum computing", 
            "artificial intelligence", "renewable energy", "blockchain",
            "cybersecurity", "space exploration", "biotechnology",
            "virtual reality"
        ]
        self.results = []
        self.session = None
    
    async def initialize_session(self):
        """Initialize the HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
    async def close_session(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    def get_random_prompt(self):
        """Generate a random prompt using templates and topics."""
        template = np.random.choice(self.prompt_templates)
        topic = np.random.choice(self.topics)
        return template.format(topic=topic)

    def get_headers(self):
        """Generate headers for API request."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def get_payload(self, prompt, max_tokens=100):
        """Generate the request payload."""
        # This is a generic format - adjust according to your API's requirements
        return {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

    async def send_request(self, prompt, max_tokens=100):
        """Send a single request to the LLM endpoint."""
        payload = self.get_payload(prompt, max_tokens)
        headers = self.get_headers()
        
        start_time = time.time()
        error = None
        
        try:
            async with self.session.post(self.endpoint_url, json=payload, headers=headers) as response:
                status_code = response.status
                if status_code == 200:
                    response_data = await response.json()
                else:
                    response_data = await response.text()
                    error = f"HTTP error {status_code}: {response_data}"
        except Exception as e:
            status_code = 0
            response_data = None
            error = str(e)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        
        return {
            "timestamp": start_time,
            "prompt": prompt,
            "status_code": status_code,
            "latency_ms": latency,
            "success": status_code == 200,
            "error": error
        }

    async def run_batch(self, batch_size, max_tokens=100):
        """Send a batch of requests concurrently."""
        await self.initialize_session()
        prompts = [self.get_random_prompt() for _ in range(batch_size)]
        tasks = [self.send_request(prompt, max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def run_load_test(self, total_requests, concurrency, ramp_up_steps=5, max_tokens=100):
        """Run a full load test with gradual ramp-up."""
        await self.initialize_session()
        print(f"\nStarting load test: {total_requests} total requests with max {concurrency} concurrent requests")
        
        # Calculate batch sizes for ramp-up
        batch_sizes = np.linspace(concurrency // ramp_up_steps, concurrency, ramp_up_steps).astype(int)
        requests_per_batch = total_requests // len(batch_sizes)
        
        for batch_size in batch_sizes:
            print(f"\nRamping up to {batch_size} concurrent requests")
            num_iterations = requests_per_batch // batch_size
            
            for _ in tqdm(range(num_iterations), desc=f"Processing batches of {batch_size}"):
                batch_results = await self.run_batch(batch_size, max_tokens)
                self.results.extend(batch_results)
                
                # Calculate and show current stats
                success_rate = sum(r["success"] for r in batch_results) / len(batch_results)
                avg_latency = np.mean([r["latency_ms"] for r in batch_results])
                
                print(f"Batch complete - Success rate: {success_rate:.2%}, Avg latency: {avg_latency:.2f}ms")
        
        await self.close_session()
        print("\nLoad test complete!")
        return self.results

    def analyze_results(self):
        """Analyze and display test results."""
        if not self.results:
            print("No results to analyze.")
            return
        
        df = pd.DataFrame(self.results)
        
        # Overall statistics
        total_requests = len(df)
        successful_requests = df["success"].sum()
        success_rate = successful_requests / total_requests
        
        # Latency statistics
        avg_latency = df["latency_ms"].mean()
        p50_latency = df["latency_ms"].quantile(0.5)
        p90_latency = df["latency_ms"].quantile(0.9)
        p95_latency = df["latency_ms"].quantile(0.95)
        p99_latency = df["latency_ms"].quantile(0.99)
        
        # Error analysis
        error_counts = df[~df["success"]]["error"].value_counts()
        
        # Results summary
        print("\n=== STRESS TEST RESULTS ===")
        print(f"Total Requests: {total_requests}")
        print(f"Successful Requests: {successful_requests}")
        print(f"Success Rate: {success_rate:.2%}")
        print("\nLatency Statistics (ms):")
        print(f"  Average: {avg_latency:.2f}")
        print(f"  50th percentile (median): {p50_latency:.2f}")
        print(f"  90th percentile: {p90_latency:.2f}")
        print(f"  95th percentile: {p95_latency:.2f}")
        print(f"  99th percentile: {p99_latency:.2f}")
        
        if len(error_counts) > 0:
            print("\nTop Errors:")
            for error, count in error_counts.head(5).items():
                print(f"  {count} occurrences: {error[:100]}...")
        
        # Calculate throughput over time
        df["timestamp_bucket"] = df["timestamp"].apply(lambda x: int(x * 2) / 2)  # 0.5 second buckets
        throughput = df.groupby("timestamp_bucket").size()
        
        # Load over time
        concurrency_over_time = df.groupby("timestamp_bucket").size() * 2  # Requests per second
        
        # Save results to CSV
        df.to_csv("stress_test_results.csv", index=False)
        print("\nDetailed results saved to 'stress_test_results.csv'")
        
        return {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "p50_latency": p50_latency,
            "p90_latency": p90_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "throughput": throughput.mean(),
            "max_concurrency": concurrency_over_time.max(),
            "df": df
        }
    
    def plot_results(self):
        """Generate plots from test results."""
        if not self.results:
            print("No results to plot.")
            return
        
        df = pd.DataFrame(self.results)
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Latency distribution
        axes[0, 0].hist(df["latency_ms"], bins=30, alpha=0.7)
        axes[0, 0].set_title("Latency Distribution")
        axes[0, 0].set_xlabel("Latency (ms)")
        axes[0, 0].set_ylabel("Frequency")
        
        # Plot 2: Latency over time
        df_sorted = df.sort_values("timestamp")
        axes[0, 1].scatter(range(len(df_sorted)), df_sorted["latency_ms"], alpha=0.5, s=10)
        axes[0, 1].set_title("Latency Over Time")
        axes[0, 1].set_xlabel("Request Number")
        axes[0, 1].set_ylabel("Latency (ms)")
        
        # Plot 3: Success rate over time
        df_sorted["rolling_success"] = df_sorted["success"].rolling(window=100, min_periods=1).mean()
        axes[1, 0].plot(range(len(df_sorted)), df_sorted["rolling_success"] * 100)
        axes[1, 0].set_title("Success Rate Over Time (100-request rolling window)")
        axes[1, 0].set_xlabel("Request Number")
        axes[1, 0].set_ylabel("Success Rate (%)")
        axes[1, 0].set_ylim(0, 100)
        
        # Plot 4: Throughput over time
        df["timestamp_bucket"] = df["timestamp"].apply(lambda x: int(x * 2) / 2)  # 0.5 second buckets
        throughput = df.groupby("timestamp_bucket").size() * 2  # Convert to req/s
        axes[1, 1].plot(throughput.index, throughput.values)
        axes[1, 1].set_title("Throughput Over Time")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Throughput (req/s)")
        
        plt.tight_layout()
        plt.savefig("stress_test_results.png")
        print("Plots saved to 'stress_test_results.png'")

async def main():
    parser = argparse.ArgumentParser(description="LLM Inference Endpoint Stress Tester")
    parser.add_argument("--url", required=True, help="Endpoint URL for the LLM service")
    parser.add_argument("--api-key", help="API key for authentication (if required)")
    parser.add_argument("--requests", type=int, default=1000, help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate per request")
    parser.add_argument("--ramp-up-steps", type=int, default=5, help="Number of steps to ramp up to max concurrency")
    
    args = parser.parse_args()
    
    tester = LLMStressTester(args.url, args.api_key)
    results = await tester.run_load_test(
        total_requests=args.requests,
        concurrency=args.concurrency,
        ramp_up_steps=args.ramp_up_steps,
        max_tokens=args.max_tokens
    )
    
    tester.analyze_results()
    tester.plot_results()

if __name__ == "__main__":
    asyncio.run(main())