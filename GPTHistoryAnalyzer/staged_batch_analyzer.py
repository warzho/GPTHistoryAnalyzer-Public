# V12 - improved prompts for decreased repetitiveness

# !/usr/bin/env python3
"""
Staged Batch Analyzer - Handles token limits intelligently
This system processes your chat history in stages, respecting OpenAI's 5M token queue limit
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import tiktoken
from openai import OpenAI


class StagedBatchAnalyzer:
    """
    Manages batch submissions within token limits
    Think of this as a smart loading dock that knows exactly how much can fit in each truck
    """

    def __init__(self, config):
        """Initialize with your existing configuration"""
        self.config = config
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.encoding = tiktoken.encoding_for_model("gpt-4")

        # Token limit with safety buffer (90% of 5M to avoid edge cases)
        self.MAX_ENQUEUED_TOKENS = 4_500_000

        # Tracking files for staged processing
        self.progress_file = Path("batch_progress.json")
        self.results_dir = Path("staged_results")
        self.results_dir.mkdir(exist_ok=True)

    def estimate_request_tokens(self, messages: List[Dict], max_tokens: int) -> int:
        """
        Accurately estimates tokens for a single request
        This is like weighing a package before shipping
        """
        total_tokens = 0

        # Count tokens in all messages
        for message in messages:
            if isinstance(message.get('content'), str):
                total_tokens += len(self.encoding.encode(message['content']))

        # Add the maximum possible response tokens
        total_tokens += max_tokens

        # Add ~10% overhead for JSON structure and system prompts
        total_tokens = int(total_tokens * 1.1)

        return total_tokens

    def filter_conversations_by_date(self, conversations: List[Dict],
                                     months_back: int = 12) -> List[Dict]:
        """
        Filters conversations to only include those from the past N months
        This helps you focus on recent, relevant history
        """
        cutoff_date = datetime.now() - timedelta(days=months_back * 30)

        filtered_conversations = []
        for conv in conversations:
            # Get the create_time value - it might be a float (Unix timestamp) or string
            create_time = conv.get('create_time', 0)

            try:
                if isinstance(create_time, (int, float)):
                    # Unix timestamp - convert to datetime
                    conv_date = datetime.fromtimestamp(create_time)
                elif isinstance(create_time, str):
                    # ISO format string - parse it
                    # Handle both with and without 'Z' timezone indicator
                    if create_time.endswith('Z'):
                        create_time = create_time.replace('Z', '+00:00')
                    conv_date = datetime.fromisoformat(create_time)
                else:
                    # Unknown format - skip this conversation
                    print(f"  ⚠️  Skipping conversation with unknown date format: {type(create_time)}")
                    continue

                if conv_date >= cutoff_date:
                    filtered_conversations.append(conv)

            except (ValueError, TypeError, AttributeError) as e:
                print(f"  ⚠️  Could not parse date for conversation: {create_time} - {str(e)}")
                continue

        print(f"Filtered to {len(filtered_conversations)} conversations from the past {months_back} months")
        print(f"Date range: {cutoff_date.strftime('%Y-%m-%d')} to present")

        return filtered_conversations

    def create_staged_batches(self, conversations: List[Dict],
                              chunk_size: int = 50) -> List[Dict]:
        """
        Creates batches that fit within token limits
        Each batch is like a shipping container with a weight limit
        """
        # First, create chunks from conversations
        chunks = self.create_conversation_chunks(conversations, chunk_size)

        # Now group chunks into batches that fit token limits
        batches = []
        current_batch = {
            'requests': [],
            'estimated_tokens': 0,
            'chunk_ids': []
        }

        for chunk in chunks:
            # Create the three analysis requests for this chunk
            requests = self.create_analysis_requests(chunk)

            # Estimate tokens for all three requests
            chunk_tokens = 0
            for req in requests:
                tokens = self.estimate_request_tokens(
                    req['body']['messages'],
                    req['body'].get('max_completion_tokens', req['body'].get('max_tokens', 4000))
                    # Handle both parameter names
                )
                chunk_tokens += tokens

            # Check if adding this chunk would exceed limit
            if current_batch['estimated_tokens'] + chunk_tokens > self.MAX_ENQUEUED_TOKENS:
                # Save current batch and start a new one
                if current_batch['requests']:
                    batches.append(current_batch)

                current_batch = {
                    'requests': [],
                    'estimated_tokens': 0,
                    'chunk_ids': []
                }

            # Add chunk to current batch
            current_batch['requests'].extend(requests)
            current_batch['estimated_tokens'] += chunk_tokens
            current_batch['chunk_ids'].append(chunk['chunk_id'])

        # Don't forget the last batch
        if current_batch['requests']:
            batches.append(current_batch)

        print(f"\nCreated {len(batches)} staged batches:")
        for i, batch in enumerate(batches, 1):
            print(f"  Batch {i}: {len(batch['requests'])} requests, "
                  f"~{batch['estimated_tokens']:,} tokens")

        return batches

    def create_conversation_chunks(self, conversations: List[Dict],
                                   chunk_size: int) -> List[Dict]:
        """
        Chunks conversations into manageable pieces
        This is your existing chunking logic, which we'll preserve
        """
        chunks = []

        for i in range(0, len(conversations), chunk_size):
            chunk = conversations[i:i + chunk_size]

            # Create a chunk identifier
            chunk_id = f"chunk_{i // chunk_size + 1:04d}"

            chunks.append({
                'chunk_id': chunk_id,
                'conversations': chunk,
                'conversation_indices': list(range(i, min(i + chunk_size, len(conversations))))
            })

        return chunks

    def create_analysis_requests(self, chunk: Dict) -> List[Dict]:
        """
        Creates the three analysis requests for a chunk
        This preserves your existing three-pronged analysis approach
        """
        requests = []

        # Combine conversations into context
        context = self.format_conversations_for_analysis(chunk['conversations'])

        # Get the prompts - improved to reduce repetitive output
        project_prompt = (getattr(self.config, 'PROJECT_ANALYSIS_PROMPT', None) or
                          getattr(self.config, 'PROJECT_EVOLUTION_PROMPT', None) or
                          getattr(self.config, 'SYSTEM_PROMPT_PROJECT', None) or
                          """Analyze these conversations for project evolution and development patterns.
 
                          Provide a structured analysis with:
                          1. A 2-3 sentence executive summary
                          2. List of specific projects identified (name each distinctly)
                          3. Development trajectory for each project
                          4. Key milestones achieved
                          5. Patterns in how projects are approached
                          6. 3-5 specific, actionable recommendations
 
                          Avoid repeating information between sections. Be concise and specific.""")

        inquiry_prompt = (getattr(self.config, 'INQUIRY_ANALYSIS_PROMPT', None) or
                          getattr(self.config, 'INQUIRY_PATTERNS_PROMPT', None) or
                          getattr(self.config, 'SYSTEM_PROMPT_INQUIRY', None) or
                          """Analyze these conversations for patterns in questions and learning approach.
 
                          Provide a structured analysis with:
                          1. A 2-3 sentence summary of learning style
                          2. Categories of problems addressed (with counts)
                          3. Preferred interaction patterns
                          4. Specific examples of iterative learning
                          5. Knowledge gaps or areas for growth
                          6. 3-5 specific recommendations for optimization
 
                          Focus on patterns, not individual conversations. Avoid redundancy.""")

        knowledge_prompt = (getattr(self.config, 'KNOWLEDGE_EVOLUTION_PROMPT', None) or
                            getattr(self.config, 'KNOWLEDGE_ANALYSIS_PROMPT', None) or
                            getattr(self.config, 'SYSTEM_PROMPT_KNOWLEDGE', None) or
                            """Analyze these conversations for knowledge growth and expertise development.
 
                            Provide a structured analysis with:
                            1. A 2-3 sentence overview of knowledge trajectory
                            2. Domains of expertise developed (ranked by depth)
                            3. Progression from novice to advanced in specific areas
                            4. Key learning breakthroughs or "aha" moments
                            5. Cross-domain knowledge transfer examples
                            6. 3-5 specific next steps for continued growth
 
                            Emphasize evolution and growth, not just current state. Be specific.""")

        # Get max tokens - use a reasonable limit for batch processing
        max_tokens = (getattr(self.config, 'BATCH_MAX_OUTPUT_TOKENS', None) or
                      getattr(self.config, 'MAX_OUTPUT_TOKENS', 4000) if getattr(self.config, 'MAX_OUTPUT_TOKENS',
                                                                                 0) > 10000
                      else getattr(self.config, 'MAX_OUTPUT_TOKENS', None) or
                           4000)  # Default fallback

        if max_tokens > 10000:
            print(f"  ⚠️  Warning: MAX_OUTPUT_TOKENS of {max_tokens} is very high for batch processing")
            print(f"     Using 4000 instead. Add BATCH_MAX_OUTPUT_TOKENS to config to customize.")
            max_tokens = 4000

        # Your three analysis types
        analysis_types = [
            {
                'custom_id': f"project_evolution-{chunk['chunk_id']}",
                'prompt': project_prompt
            },
            {
                'custom_id': f"inquiry_patterns-{chunk['chunk_id']}",
                'prompt': inquiry_prompt
            },
            {
                'custom_id': f"knowledge_evolution-{chunk['chunk_id']}",
                'prompt': knowledge_prompt
            }
        ]

        # Get the model name from config
        model_name = getattr(self.config, 'ANALYSIS_MODEL', 'gpt-4o-mini')

        # Temporary override for testing - REMOVE THIS after confirming it works
        # model_name = 'gpt-4o-mini'  # Uncomment to test with standard model

        for analysis in analysis_types:
            request = {
                "custom_id": analysis['custom_id'],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": analysis['prompt']},
                        {"role": "user", "content": context}
                    ],
                    "max_completion_tokens": max_tokens,  # Changed from max_tokens
                    "temperature": 1
                }
            }
            requests.append(request)

        return requests

    def format_conversations_for_analysis(self, conversations: List[Dict]) -> str:
        """
        Formats conversations for analysis
        This is where you'd apply any preprocessing or formatting
        """
        formatted_text = ""

        for conv in conversations:
            formatted_text += f"\n\n--- Conversation ---\n"
            formatted_text += f"Date: {conv.get('create_time', 'Unknown')}\n"
            formatted_text += f"Title: {conv.get('title', 'Untitled')}\n\n"

            # Add the actual conversation content
            # Adjust this based on your data structure
            for node_id, node_data in conv.get('mapping', {}).items():
                message = node_data.get('message')
                if message and message.get('content'):
                    role = message.get('author', {}).get('role', 'unknown')
                    content = message.get('content', {}).get('parts', [''])[0]
                    formatted_text += f"{role.upper()}: {content}\n\n"

        return formatted_text

    def submit_staged_batches(self, conversations: List[Dict], months_back: int = 12) -> Dict:
        """
        Main orchestration function - submits batches in stages
        This is the conductor that manages the entire process
        """
        print("\n🚀 Starting Staged Batch Submission Process")
        print("=" * 60)

        # Filter to specified months
        filtered_convs = self.filter_conversations_by_date(conversations, months_back=months_back)

        # Create staged batches
        batches = self.create_staged_batches(filtered_convs)

        # Load or create progress tracking
        progress = self.load_progress()

        # Process each batch
        for batch_num, batch in enumerate(batches, 1):
            print(f"\n📦 Processing Batch {batch_num} of {len(batches)}")
            print(f"   Chunks: {', '.join(batch['chunk_ids'])}")
            print(f"   Estimated tokens: {batch['estimated_tokens']:,}")

            # Check if this batch was already processed
            if self.is_batch_completed(batch_num, progress):
                print(f"   ✅ Already completed, skipping...")
                continue

            # Submit the batch
            batch_id = self.submit_single_batch(batch, batch_num)

            if batch_id:
                # Update progress
                progress['batches'][batch_num] = {
                    'batch_id': batch_id,
                    'status': 'submitted',
                    'submitted_at': datetime.now().isoformat(),
                    'chunk_ids': batch['chunk_ids']
                }
                self.save_progress(progress)

                # Wait for completion before submitting next batch
                print(f"   ⏳ Waiting for batch to complete...")
                self.wait_for_batch_completion(batch_id, batch_num, progress)
            else:
                print(f"   ❌ Failed to submit batch {batch_num}")
                return {'success': False, 'error': f'Failed at batch {batch_num}'}

        print("\n✨ All batches completed successfully!")
        return {'success': True, 'progress': progress}

    def submit_single_batch(self, batch: Dict, batch_num: int) -> Optional[str]:
        """
        Submits a single batch to OpenAI
        Returns the batch ID if successful
        """
        try:
            # Write requests to JSONL file
            jsonl_path = self.results_dir / f"batch_{batch_num:03d}_requests.jsonl"

            with open(jsonl_path, 'w') as f:
                for request in batch['requests']:
                    f.write(json.dumps(request) + '\n')

            # Upload file to OpenAI
            with open(jsonl_path, 'rb') as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose='batch'
                )

            # Create batch job
            batch_response = self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )

            print(f"   ✅ Submitted batch ID: {batch_response.id}")
            return batch_response.id

        except Exception as e:
            print(f"   ❌ Error submitting batch: {str(e)}")
            return None

    def wait_for_batch_completion(self, batch_id: str, batch_num: int,
                                  progress: Dict, check_interval: int = 300):
        """
        Waits for a batch to complete, with periodic status updates
        check_interval is in seconds (default 5 minutes)
        """
        start_time = time.time()

        while True:
            try:
                batch = self.client.batches.retrieve(batch_id)

                if batch.status == 'completed':
                    print(f"   ✅ Batch {batch_num} completed!")

                    # Retrieve and save results
                    self.retrieve_batch_results(batch_id, batch_num)

                    # Update progress
                    progress['batches'][batch_num]['status'] = 'completed'
                    progress['batches'][batch_num]['completed_at'] = datetime.now().isoformat()
                    self.save_progress(progress)

                    return True

                elif batch.status == 'failed':
                    print(f"   ❌ Batch {batch_num} failed!")
                    progress['batches'][batch_num]['status'] = 'failed'
                    self.save_progress(progress)
                    return False

                else:
                    # Still processing
                    elapsed = (time.time() - start_time) / 60
                    if batch.request_counts:
                        completed = batch.request_counts.completed
                        total = batch.request_counts.total
                        print(f"   ⏳ Status: {batch.status} | Progress: {completed}/{total} | "
                              f"Elapsed: {elapsed:.1f} minutes")

                    time.sleep(check_interval)

            except Exception as e:
                print(f"   ⚠️ Error checking status: {str(e)}")
                time.sleep(check_interval)

    def retrieve_batch_results(self, batch_id: str, batch_num: int):
        """
        Downloads and saves results from a completed batch
        """
        try:
            batch = self.client.batches.retrieve(batch_id)

            if batch.output_file_id:
                # Download results
                content = self.client.files.content(batch.output_file_id)

                # Save to file
                results_path = self.results_dir / f"batch_{batch_num:03d}_results.jsonl"
                results_path.write_bytes(content.read())

                print(f"   💾 Results saved to {results_path}")

        except Exception as e:
            print(f"   ⚠️ Error retrieving results: {str(e)}")

    def load_progress(self) -> Dict:
        """Loads progress tracking from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'batches': {}, 'started_at': datetime.now().isoformat()}

    def save_progress(self, progress: Dict):
        """Saves progress tracking to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def is_batch_completed(self, batch_num: int, progress: Dict) -> bool:
        """Checks if a batch was already completed"""
        batch_info = progress.get('batches', {}).get(str(batch_num), {})
        return batch_info.get('status') == 'completed'

    def combine_all_results(self) -> Dict:
        """
        Combines results from all completed batches into final analysis
        This is like assembling puzzle pieces into the complete picture
        """
        combined_results = {
            'project_evolution': [],
            'inquiry_patterns': [],
            'knowledge_evolution': []
        }

        # Read all result files
        result_files = sorted(self.results_dir.glob("batch_*_results.jsonl"))

        for result_file in result_files:
            with open(result_file, 'r') as f:
                for line in f:
                    result = json.loads(line)

                    # Categorize by analysis type
                    custom_id = result.get('custom_id', '')

                    if 'project_evolution' in custom_id:
                        combined_results['project_evolution'].append(result)
                    elif 'inquiry_patterns' in custom_id:
                        combined_results['inquiry_patterns'].append(result)
                    elif 'knowledge_evolution' in custom_id:
                        combined_results['knowledge_evolution'].append(result)

        # Save combined results
        combined_path = self.results_dir / "combined_analysis.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_results, f, indent=2)

        print(f"\n📊 Combined results saved to {combined_path}")
        print(f"   Project Evolution: {len(combined_results['project_evolution'])} analyses")
        print(f"   Inquiry Patterns: {len(combined_results['inquiry_patterns'])} analyses")
        print(f"   Knowledge Evolution: {len(combined_results['knowledge_evolution'])} analyses")

        return combined_results

#Version 8

# # !/usr/bin/env python3
# """
# Staged Batch Analyzer - Handles token limits intelligently
# This system processes your chat history in stages, respecting OpenAI's 5M token queue limit
# """
#
# import json
# import time
# import sys
# from pathlib import Path
# from datetime import datetime, timedelta
# from typing import List, Dict, Optional, Tuple
# import tiktoken
# from openai import OpenAI
#
#
# class StagedBatchAnalyzer:
#     """
#     Manages batch submissions within token limits
#     Think of this as a smart loading dock that knows exactly how much can fit in each truck
#     """
#
#     def __init__(self, config):
#         """Initialize with your existing configuration"""
#         self.config = config
#         self.client = OpenAI(api_key=config.OPENAI_API_KEY)
#         self.encoding = tiktoken.encoding_for_model("gpt-4")
#
#         # Token limit with safety buffer (90% of 5M to avoid edge cases)
#         self.MAX_ENQUEUED_TOKENS = 4_500_000
#
#         # Tracking files for staged processing
#         self.progress_file = Path("batch_progress.json")
#         self.results_dir = Path("staged_results")
#         self.results_dir.mkdir(exist_ok=True)
#
#     def estimate_request_tokens(self, messages: List[Dict], max_tokens: int) -> int:
#         """
#         Accurately estimates tokens for a single request
#         This is like weighing a package before shipping
#         """
#         total_tokens = 0
#
#         # Count tokens in all messages
#         for message in messages:
#             if isinstance(message.get('content'), str):
#                 total_tokens += len(self.encoding.encode(message['content']))
#
#         # Add the maximum possible response tokens
#         total_tokens += max_tokens
#
#         # Add ~10% overhead for JSON structure and system prompts
#         total_tokens = int(total_tokens * 1.1)
#
#         return total_tokens
#
#     def filter_conversations_by_date(self, conversations: List[Dict],
#                                      months_back: int = 12) -> List[Dict]:
#         """
#         Filters conversations to only include those from the past N months
#         This helps you focus on recent, relevant history
#         """
#         cutoff_date = datetime.now() - timedelta(days=months_back * 30)
#
#         filtered_conversations = []
#         for conv in conversations:
#             # Get the create_time value - it might be a float (Unix timestamp) or string
#             create_time = conv.get('create_time', 0)
#
#             try:
#                 if isinstance(create_time, (int, float)):
#                     # Unix timestamp - convert to datetime
#                     conv_date = datetime.fromtimestamp(create_time)
#                 elif isinstance(create_time, str):
#                     # ISO format string - parse it
#                     # Handle both with and without 'Z' timezone indicator
#                     if create_time.endswith('Z'):
#                         create_time = create_time.replace('Z', '+00:00')
#                     conv_date = datetime.fromisoformat(create_time)
#                 else:
#                     # Unknown format - skip this conversation
#                     print(f"  ⚠️  Skipping conversation with unknown date format: {type(create_time)}")
#                     continue
#
#                 if conv_date >= cutoff_date:
#                     filtered_conversations.append(conv)
#
#             except (ValueError, TypeError, AttributeError) as e:
#                 print(f"  ⚠️  Could not parse date for conversation: {create_time} - {str(e)}")
#                 continue
#
#         print(f"Filtered to {len(filtered_conversations)} conversations from the past {months_back} months")
#         print(f"Date range: {cutoff_date.strftime('%Y-%m-%d')} to present")
#
#         return filtered_conversations
#
#     def create_staged_batches(self, conversations: List[Dict],
#                               chunk_size: int = 50) -> List[Dict]:
#         """
#         Creates batches that fit within token limits
#         Each batch is like a shipping container with a weight limit
#         """
#         # First, create chunks from conversations
#         chunks = self.create_conversation_chunks(conversations, chunk_size)
#
#         # Now group chunks into batches that fit token limits
#         batches = []
#         current_batch = {
#             'requests': [],
#             'estimated_tokens': 0,
#             'chunk_ids': []
#         }
#
#         for chunk in chunks:
#             # Create the three analysis requests for this chunk
#             requests = self.create_analysis_requests(chunk)
#
#             # Estimate tokens for all three requests
#             chunk_tokens = 0
#             for req in requests:
#                 #Adjusting for gpt-5-mini max completion tokens
#                 tokens = self.estimate_request_tokens(
#                     req['body']['messages'],
#                     req['body'].get('max_completion_tokens', req['body'].get('max_tokens', 4000))
#                 )
#                 chunk_tokens += tokens
#
#             # Check if adding this chunk would exceed limit
#             if current_batch['estimated_tokens'] + chunk_tokens > self.MAX_ENQUEUED_TOKENS:
#                 # Save current batch and start a new one
#                 if current_batch['requests']:
#                     batches.append(current_batch)
#
#                 current_batch = {
#                     'requests': [],
#                     'estimated_tokens': 0,
#                     'chunk_ids': []
#                 }
#
#             # Add chunk to current batch
#             current_batch['requests'].extend(requests)
#             current_batch['estimated_tokens'] += chunk_tokens
#             current_batch['chunk_ids'].append(chunk['chunk_id'])
#
#         # Don't forget the last batch
#         if current_batch['requests']:
#             batches.append(current_batch)
#
#         print(f"\nCreated {len(batches)} staged batches:")
#         for i, batch in enumerate(batches, 1):
#             print(f"  Batch {i}: {len(batch['requests'])} requests, "
#                   f"~{batch['estimated_tokens']:,} tokens")
#
#         return batches
#
#     def create_conversation_chunks(self, conversations: List[Dict],
#                                    chunk_size: int) -> List[Dict]:
#         """
#         Chunks conversations into manageable pieces
#         This is your existing chunking logic, which we'll preserve
#         """
#         chunks = []
#
#         for i in range(0, len(conversations), chunk_size):
#             chunk = conversations[i:i + chunk_size]
#
#             # Create a chunk identifier
#             chunk_id = f"chunk_{i // chunk_size + 1:04d}"
#
#             chunks.append({
#                 'chunk_id': chunk_id,
#                 'conversations': chunk,
#                 'conversation_indices': list(range(i, min(i + chunk_size, len(conversations))))
#             })
#
#         return chunks
#
#     def create_analysis_requests(self, chunk: Dict) -> List[Dict]:
#         """
#         Creates the three analysis requests for a chunk
#         This preserves your existing three-pronged analysis approach
#         """
#         requests = []
#
#         # Combine conversations into context
#         context = self.format_conversations_for_analysis(chunk['conversations'])
#
#         # Get the prompts - check for different possible attribute names
#         # Your config might use different names, so we'll check for common variations
#         project_prompt = (getattr(self.config, 'PROJECT_ANALYSIS_PROMPT', None) or
#                           getattr(self.config, 'PROJECT_EVOLUTION_PROMPT', None) or
#                           getattr(self.config, 'SYSTEM_PROMPT_PROJECT', None) or
#                           """Analyze these conversations for project evolution and development patterns.
#                           Identify key projects discussed, their progression over time, and major milestones.""")
#
#         inquiry_prompt = (getattr(self.config, 'INQUIRY_ANALYSIS_PROMPT', None) or
#                           getattr(self.config, 'INQUIRY_PATTERNS_PROMPT', None) or
#                           getattr(self.config, 'SYSTEM_PROMPT_INQUIRY', None) or
#                           """Analyze these conversations for patterns in questions and inquiries.
#                           Identify recurring themes, types of problems solved, and learning patterns.""")
#
#         knowledge_prompt = (getattr(self.config, 'KNOWLEDGE_EVOLUTION_PROMPT', None) or
#                             getattr(self.config, 'KNOWLEDGE_ANALYSIS_PROMPT', None) or
#                             getattr(self.config, 'SYSTEM_PROMPT_KNOWLEDGE', None) or
#                             """Analyze these conversations for knowledge evolution and growth.
#                             Track how understanding deepened over time and identify key learning moments.""")
#
#         # Get max tokens - use a reasonable limit for batch processing
#         max_tokens = (getattr(self.config, 'BATCH_MAX_OUTPUT_TOKENS', None) or
#                       getattr(self.config, 'MAX_OUTPUT_TOKENS', 4000) if getattr(self.config, 'MAX_OUTPUT_TOKENS',
#                                                                                  0) > 10000
#                       else getattr(self.config, 'MAX_OUTPUT_TOKENS', None) or
#                            4000)  # Default fallback
#
#         if max_tokens > 10000:
#             print(f"  ⚠️  Warning: MAX_OUTPUT_TOKENS of {max_tokens} is very high for batch processing")
#             print(f"     Using 4000 instead. Add BATCH_MAX_OUTPUT_TOKENS to config to customize.")
#             max_tokens = 4000
#
#         # Your three analysis types
#         analysis_types = [
#             {
#                 'custom_id': f"project_evolution-{chunk['chunk_id']}",
#                 'prompt': project_prompt
#             },
#             {
#                 'custom_id': f"inquiry_patterns-{chunk['chunk_id']}",
#                 'prompt': inquiry_prompt
#             },
#             {
#                 'custom_id': f"knowledge_evolution-{chunk['chunk_id']}",
#                 'prompt': knowledge_prompt
#             }
#         ]
#
#         # Get the model name from config
#         model_name = getattr(self.config, 'ANALYSIS_MODEL', 'gpt-5-mini')
#
#         for analysis in analysis_types:
#             request = {
#                 "custom_id": analysis['custom_id'],
#                 "method": "POST",
#                 "url": "/v1/chat/completions",
#                 "body": {
#                     "model": model_name,
#                     "messages": [
#                         {"role": "system", "content": analysis['prompt']},
#                         {"role": "user", "content": context}
#                     ],
#                     #Adjusting for gpt-5-mini
#                     "max_completion_tokens": max_tokens,
#                     "temperature": 1
#                 }
#             }
#             requests.append(request)
#
#         return requests
#
#     def format_conversations_for_analysis(self, conversations: List[Dict]) -> str:
#         """
#         Formats conversations for analysis
#         This is where you'd apply any preprocessing or formatting
#         """
#         formatted_text = ""
#
#         for conv in conversations:
#             formatted_text += f"\n\n--- Conversation ---\n"
#             formatted_text += f"Date: {conv.get('create_time', 'Unknown')}\n"
#             formatted_text += f"Title: {conv.get('title', 'Untitled')}\n\n"
#
#             # Add the actual conversation content
#             # Adjust this based on your data structure
#             for node_id, node_data in conv.get('mapping', {}).items():
#                 message = node_data.get('message')
#                 if message and message.get('content'):
#                     role = message.get('author', {}).get('role', 'unknown')
#                     content = message.get('content', {}).get('parts', [''])[0]
#                     formatted_text += f"{role.upper()}: {content}\n\n"
#
#         return formatted_text
#
#     def submit_staged_batches(self, conversations: List[Dict], months_back: int = 12) -> Dict:
#         """
#         Main orchestration function - submits batches in stages
#         This is the conductor that manages the entire process
#         """
#         print("\n🚀 Starting Staged Batch Submission Process")
#         print("=" * 60)
#
#         # Filter to specified months
#         filtered_convs = self.filter_conversations_by_date(conversations, months_back=months_back)
#
#         # Create staged batches
#         batches = self.create_staged_batches(filtered_convs)
#
#         # Load or create progress tracking
#         progress = self.load_progress()
#
#         # Process each batch
#         for batch_num, batch in enumerate(batches, 1):
#             print(f"\n📦 Processing Batch {batch_num} of {len(batches)}")
#             print(f"   Chunks: {', '.join(batch['chunk_ids'])}")
#             print(f"   Estimated tokens: {batch['estimated_tokens']:,}")
#
#             # Check if this batch was already processed
#             if self.is_batch_completed(batch_num, progress):
#                 print(f"   ✅ Already completed, skipping...")
#                 continue
#
#             # Submit the batch
#             batch_id = self.submit_single_batch(batch, batch_num)
#
#             if batch_id:
#                 # Update progress
#                 progress['batches'][batch_num] = {
#                     'batch_id': batch_id,
#                     'status': 'submitted',
#                     'submitted_at': datetime.now().isoformat(),
#                     'chunk_ids': batch['chunk_ids']
#                 }
#                 self.save_progress(progress)
#
#                 # Wait for completion before submitting next batch
#                 print(f"   ⏳ Waiting for batch to complete...")
#                 self.wait_for_batch_completion(batch_id, batch_num, progress)
#             else:
#                 print(f"   ❌ Failed to submit batch {batch_num}")
#                 return {'success': False, 'error': f'Failed at batch {batch_num}'}
#
#         print("\n✨ All batches completed successfully!")
#         return {'success': True, 'progress': progress}
#
#     def submit_single_batch(self, batch: Dict, batch_num: int) -> Optional[str]:
#         """
#         Submits a single batch to OpenAI
#         Returns the batch ID if successful
#         """
#         try:
#             # Write requests to JSONL file
#             jsonl_path = self.results_dir / f"batch_{batch_num:03d}_requests.jsonl"
#
#             with open(jsonl_path, 'w') as f:
#                 for request in batch['requests']:
#                     f.write(json.dumps(request) + '\n')
#
#             # Upload file to OpenAI
#             with open(jsonl_path, 'rb') as f:
#                 file_response = self.client.files.create(
#                     file=f,
#                     purpose='batch'
#                 )
#
#             # Create batch job
#             batch_response = self.client.batches.create(
#                 input_file_id=file_response.id,
#                 endpoint="/v1/chat/completions",
#                 completion_window="24h"
#             )
#
#             print(f"   ✅ Submitted batch ID: {batch_response.id}")
#             return batch_response.id
#
#         except Exception as e:
#             print(f"   ❌ Error submitting batch: {str(e)}")
#             return None
#
#     def wait_for_batch_completion(self, batch_id: str, batch_num: int,
#                                   progress: Dict, check_interval: int = 300):
#         """
#         Waits for a batch to complete, with periodic status updates
#         check_interval is in seconds (default 5 minutes)
#         """
#         start_time = time.time()
#
#         while True:
#             try:
#                 batch = self.client.batches.retrieve(batch_id)
#
#                 if batch.status == 'completed':
#                     print(f"   ✅ Batch {batch_num} completed!")
#
#                     # Retrieve and save results
#                     self.retrieve_batch_results(batch_id, batch_num)
#
#                     # Update progress
#                     progress['batches'][batch_num]['status'] = 'completed'
#                     progress['batches'][batch_num]['completed_at'] = datetime.now().isoformat()
#                     self.save_progress(progress)
#
#                     return True
#
#                 elif batch.status == 'failed':
#                     print(f"   ❌ Batch {batch_num} failed!")
#                     progress['batches'][batch_num]['status'] = 'failed'
#                     self.save_progress(progress)
#                     return False
#
#                 else:
#                     # Still processing
#                     elapsed = (time.time() - start_time) / 60
#                     if batch.request_counts:
#                         completed = batch.request_counts.completed
#                         total = batch.request_counts.total
#                         print(f"   ⏳ Status: {batch.status} | Progress: {completed}/{total} | "
#                               f"Elapsed: {elapsed:.1f} minutes")
#
#                     time.sleep(check_interval)
#
#             except Exception as e:
#                 print(f"   ⚠️ Error checking status: {str(e)}")
#                 time.sleep(check_interval)
#
#     def retrieve_batch_results(self, batch_id: str, batch_num: int):
#         """
#         Downloads and saves results from a completed batch
#         """
#         try:
#             batch = self.client.batches.retrieve(batch_id)
#
#             if batch.output_file_id:
#                 # Download results
#                 content = self.client.files.content(batch.output_file_id)
#
#                 # Save to file
#                 results_path = self.results_dir / f"batch_{batch_num:03d}_results.jsonl"
#                 results_path.write_bytes(content.read())
#
#                 print(f"   💾 Results saved to {results_path}")
#
#         except Exception as e:
#             print(f"   ⚠️ Error retrieving results: {str(e)}")
#
#     def load_progress(self) -> Dict:
#         """Loads progress tracking from file"""
#         if self.progress_file.exists():
#             with open(self.progress_file, 'r') as f:
#                 return json.load(f)
#         return {'batches': {}, 'started_at': datetime.now().isoformat()}
#
#     def save_progress(self, progress: Dict):
#         """Saves progress tracking to file"""
#         with open(self.progress_file, 'w') as f:
#             json.dump(progress, f, indent=2)
#
#     def is_batch_completed(self, batch_num: int, progress: Dict) -> bool:
#         """Checks if a batch was already completed"""
#         batch_info = progress.get('batches', {}).get(str(batch_num), {})
#         return batch_info.get('status') == 'completed'
#
#     def combine_all_results(self) -> Dict:
#         """
#         Combines results from all completed batches into final analysis
#         This is like assembling puzzle pieces into the complete picture
#         """
#         combined_results = {
#             'project_evolution': [],
#             'inquiry_patterns': [],
#             'knowledge_evolution': []
#         }
#
#         # Read all result files
#         result_files = sorted(self.results_dir.glob("batch_*_results.jsonl"))
#
#         for result_file in result_files:
#             with open(result_file, 'r') as f:
#                 for line in f:
#                     result = json.loads(line)
#
#                     # Categorize by analysis type
#                     custom_id = result.get('custom_id', '')
#
#                     if 'project_evolution' in custom_id:
#                         combined_results['project_evolution'].append(result)
#                     elif 'inquiry_patterns' in custom_id:
#                         combined_results['inquiry_patterns'].append(result)
#                     elif 'knowledge_evolution' in custom_id:
#                         combined_results['knowledge_evolution'].append(result)
#
#         # Save combined results
#         combined_path = self.results_dir / "combined_analysis.json"
#         with open(combined_path, 'w') as f:
#             json.dump(combined_results, f, indent=2)
#
#         print(f"\n📊 Combined results saved to {combined_path}")
#         print(f"   Project Evolution: {len(combined_results['project_evolution'])} analyses")
#         print(f"   Inquiry Patterns: {len(combined_results['inquiry_patterns'])} analyses")
#         print(f"   Knowledge Evolution: {len(combined_results['knowledge_evolution'])} analyses")
#
#         return combined_results


#Version 5

#!/usr/bin/env python3
# """
# Staged Batch Analyzer - Handles token limits intelligently
# This system processes your chat history in stages, respecting OpenAI's 5M token queue limit
# """
#
# import json
# import time
# import sys
# from pathlib import Path
# from datetime import datetime, timedelta
# from typing import List, Dict, Optional, Tuple
# import tiktoken
# from openai import OpenAI
#
#
# class StagedBatchAnalyzer:
#     """
#     Manages batch submissions within token limits
#     Think of this as a smart loading dock that knows exactly how much can fit in each truck
#     """
#
#     def __init__(self, config):
#         """Initialize with your existing configuration"""
#         self.config = config
#         self.client = OpenAI(api_key=config.OPENAI_API_KEY)
#         self.encoding = tiktoken.encoding_for_model("gpt-4")
#
#         # Token limit with safety buffer (90% of 5M to avoid edge cases)
#         self.MAX_ENQUEUED_TOKENS = 4_500_000
#
#         # Tracking files for staged processing
#         self.progress_file = Path("batch_progress.json")
#         self.results_dir = Path("staged_results")
#         self.results_dir.mkdir(exist_ok=True)
#
#     def estimate_request_tokens(self, messages: List[Dict], max_tokens: int) -> int:
#         """
#         Accurately estimates tokens for a single request
#         This is like weighing a package before shipping
#         """
#         total_tokens = 0
#
#         # Count tokens in all messages
#         for message in messages:
#             if isinstance(message.get('content'), str):
#                 total_tokens += len(self.encoding.encode(message['content']))
#
#         # Add the maximum possible response tokens
#         total_tokens += max_tokens
#
#         # Add ~10% overhead for JSON structure and system prompts
#         total_tokens = int(total_tokens * 1.1)
#
#         return total_tokens
#
#     def filter_conversations_by_date(self, conversations: List[Dict],
#                                      months_back: int = 12) -> List[Dict]:
#         """
#         Filters conversations to only include those from the past N months
#         This helps you focus on recent, relevant history
#         """
#         cutoff_date = datetime.now() - timedelta(days=months_back * 30)
#
#         filtered_conversations = []
#         for conv in conversations:
#             # Get the create_time value - it might be a float (Unix timestamp) or string
#             create_time = conv.get('create_time', 0)
#
#             try:
#                 if isinstance(create_time, (int, float)):
#                     # Unix timestamp - convert to datetime
#                     conv_date = datetime.fromtimestamp(create_time)
#                 elif isinstance(create_time, str):
#                     # ISO format string - parse it
#                     # Handle both with and without 'Z' timezone indicator
#                     if create_time.endswith('Z'):
#                         create_time = create_time.replace('Z', '+00:00')
#                     conv_date = datetime.fromisoformat(create_time)
#                 else:
#                     # Unknown format - skip this conversation
#                     print(f"  ⚠️  Skipping conversation with unknown date format: {type(create_time)}")
#                     continue
#
#                 if conv_date >= cutoff_date:
#                     filtered_conversations.append(conv)
#
#             except (ValueError, TypeError, AttributeError) as e:
#                 print(f"  ⚠️  Could not parse date for conversation: {create_time} - {str(e)}")
#                 continue
#
#         print(f"Filtered to {len(filtered_conversations)} conversations from the past {months_back} months")
#         print(f"Date range: {cutoff_date.strftime('%Y-%m-%d')} to present")
#
#         return filtered_conversations
#
#     def create_staged_batches(self, conversations: List[Dict],
#                               chunk_size: int = 50) -> List[Dict]:
#         """
#         Creates batches that fit within token limits
#         Each batch is like a shipping container with a weight limit
#         """
#         # First, create chunks from conversations
#         chunks = self.create_conversation_chunks(conversations, chunk_size)
#
#         # Now group chunks into batches that fit token limits
#         batches = []
#         current_batch = {
#             'requests': [],
#             'estimated_tokens': 0,
#             'chunk_ids': []
#         }
#
#         for chunk in chunks:
#             # Create the three analysis requests for this chunk
#             requests = self.create_analysis_requests(chunk)
#
#             # Estimate tokens for all three requests
#             chunk_tokens = 0
#             for req in requests:
#                 tokens = self.estimate_request_tokens(
#                     req['body']['messages'],
#                     req['body']['max_tokens']
#                 )
#                 chunk_tokens += tokens
#
#             # Check if adding this chunk would exceed limit
#             if current_batch['estimated_tokens'] + chunk_tokens > self.MAX_ENQUEUED_TOKENS:
#                 # Save current batch and start a new one
#                 if current_batch['requests']:
#                     batches.append(current_batch)
#
#                 current_batch = {
#                     'requests': [],
#                     'estimated_tokens': 0,
#                     'chunk_ids': []
#                 }
#
#             # Add chunk to current batch
#             current_batch['requests'].extend(requests)
#             current_batch['estimated_tokens'] += chunk_tokens
#             current_batch['chunk_ids'].append(chunk['chunk_id'])
#
#         # Don't forget the last batch
#         if current_batch['requests']:
#             batches.append(current_batch)
#
#         print(f"\nCreated {len(batches)} staged batches:")
#         for i, batch in enumerate(batches, 1):
#             print(f"  Batch {i}: {len(batch['requests'])} requests, "
#                   f"~{batch['estimated_tokens']:,} tokens")
#
#         return batches
#
#     def create_conversation_chunks(self, conversations: List[Dict],
#                                    chunk_size: int) -> List[Dict]:
#         """
#         Chunks conversations into manageable pieces
#         This is your existing chunking logic, which we'll preserve
#         """
#         chunks = []
#
#         for i in range(0, len(conversations), chunk_size):
#             chunk = conversations[i:i + chunk_size]
#
#             # Create a chunk identifier
#             chunk_id = f"chunk_{i // chunk_size + 1:04d}"
#
#             chunks.append({
#                 'chunk_id': chunk_id,
#                 'conversations': chunk,
#                 'conversation_indices': list(range(i, min(i + chunk_size, len(conversations))))
#             })
#
#         return chunks
#
#     def create_analysis_requests(self, chunk: Dict) -> List[Dict]:
#         """
#         Creates the three analysis requests for a chunk
#         This preserves your existing three-pronged analysis approach
#         """
#         requests = []
#
#         # Combine conversations into context
#         context = self.format_conversations_for_analysis(chunk['conversations'])
#
#         # Get the prompts - check for different possible attribute names
#         # Your config might use different names, so we'll check for common variations
#         project_prompt = (getattr(self.config, 'PROJECT_ANALYSIS_PROMPT', None) or
#                           getattr(self.config, 'PROJECT_EVOLUTION_PROMPT', None) or
#                           getattr(self.config, 'SYSTEM_PROMPT_PROJECT', None) or
#                           """Analyze these conversations for project evolution and development patterns.
#                           Identify key projects discussed, their progression over time, and major milestones.""")
#
#         inquiry_prompt = (getattr(self.config, 'INQUIRY_ANALYSIS_PROMPT', None) or
#                           getattr(self.config, 'INQUIRY_PATTERNS_PROMPT', None) or
#                           getattr(self.config, 'SYSTEM_PROMPT_INQUIRY', None) or
#                           """Analyze these conversations for patterns in questions and inquiries.
#                           Identify recurring themes, types of problems solved, and learning patterns.""")
#
#         knowledge_prompt = (getattr(self.config, 'KNOWLEDGE_EVOLUTION_PROMPT', None) or
#                             getattr(self.config, 'KNOWLEDGE_ANALYSIS_PROMPT', None) or
#                             getattr(self.config, 'SYSTEM_PROMPT_KNOWLEDGE', None) or
#                             """Analyze these conversations for knowledge evolution and growth.
#                             Track how understanding deepened over time and identify key learning moments.""")
#
#         # Get max tokens - your config uses MAX_OUTPUT_TOKENS
#         max_tokens = (getattr(self.config, 'MAX_OUTPUT_TOKENS', None) or
#                       getattr(self.config, 'MAX_TOKENS_PER_REQUEST', None) or
#                       getattr(self.config, 'MAX_TOKENS', None) or
#                       4000)  # Default fallback
#
#         # Your three analysis types
#         analysis_types = [
#             {
#                 'custom_id': f"project_evolution-{chunk['chunk_id']}",
#                 'prompt': project_prompt
#             },
#             {
#                 'custom_id': f"inquiry_patterns-{chunk['chunk_id']}",
#                 'prompt': inquiry_prompt
#             },
#             {
#                 'custom_id': f"knowledge_evolution-{chunk['chunk_id']}",
#                 'prompt': knowledge_prompt
#             }
#         ]
#
#         for analysis in analysis_types:
#             request = {
#                 "custom_id": analysis['custom_id'],
#                 "method": "POST",
#                 "url": "/v1/chat/completions",
#                 "body": {
#                     "model": "gpt-5-mini",
#                     "messages": [
#                         {"role": "system", "content": analysis['prompt']},
#                         {"role": "user", "content": context}
#                     ],
#                     "max_tokens": max_tokens,
#                     "temperature": 0.7
#                 }
#             }
#             requests.append(request)
#
#         return requests
#
#     def format_conversations_for_analysis(self, conversations: List[Dict]) -> str:
#         """
#         Formats conversations for analysis
#         This is where you'd apply any preprocessing or formatting
#         """
#         formatted_text = ""
#
#         for conv in conversations:
#             formatted_text += f"\n\n--- Conversation ---\n"
#             formatted_text += f"Date: {conv.get('create_time', 'Unknown')}\n"
#             formatted_text += f"Title: {conv.get('title', 'Untitled')}\n\n"
#
#             # Add the actual conversation content
#             # Adjust this based on your data structure
#             for node_id, node_data in conv.get('mapping', {}).items():
#                 message = node_data.get('message')
#                 if message and message.get('content'):
#                     role = message.get('author', {}).get('role', 'unknown')
#                     content = message.get('content', {}).get('parts', [''])[0]
#                     formatted_text += f"{role.upper()}: {content}\n\n"
#
#         return formatted_text
#
#     def submit_staged_batches(self, conversations: List[Dict]) -> Dict:
#         """
#         Main orchestration function - submits batches in stages
#         This is the conductor that manages the entire process
#         """
#         print("\n🚀 Starting Staged Batch Submission Process")
#         print("=" * 60)
#
#         # Filter to past year if specified
#         filtered_convs = self.filter_conversations_by_date(conversations, months_back=12)
#
#         # Create staged batches
#         batches = self.create_staged_batches(filtered_convs)
#
#         # Load or create progress tracking
#         progress = self.load_progress()
#
#         # Process each batch
#         for batch_num, batch in enumerate(batches, 1):
#             print(f"\n📦 Processing Batch {batch_num} of {len(batches)}")
#             print(f"   Chunks: {', '.join(batch['chunk_ids'])}")
#             print(f"   Estimated tokens: {batch['estimated_tokens']:,}")
#
#             # Check if this batch was already processed
#             if self.is_batch_completed(batch_num, progress):
#                 print(f"   ✅ Already completed, skipping...")
#                 continue
#
#             # Submit the batch
#             batch_id = self.submit_single_batch(batch, batch_num)
#
#             if batch_id:
#                 # Update progress
#                 progress['batches'][batch_num] = {
#                     'batch_id': batch_id,
#                     'status': 'submitted',
#                     'submitted_at': datetime.now().isoformat(),
#                     'chunk_ids': batch['chunk_ids']
#                 }
#                 self.save_progress(progress)
#
#                 # Wait for completion before submitting next batch
#                 print(f"   ⏳ Waiting for batch to complete...")
#                 self.wait_for_batch_completion(batch_id, batch_num, progress)
#             else:
#                 print(f"   ❌ Failed to submit batch {batch_num}")
#                 return {'success': False, 'error': f'Failed at batch {batch_num}'}
#
#         print("\n✨ All batches completed successfully!")
#         return {'success': True, 'progress': progress}
#
#     def submit_single_batch(self, batch: Dict, batch_num: int) -> Optional[str]:
#         """
#         Submits a single batch to OpenAI
#         Returns the batch ID if successful
#         """
#         try:
#             # Write requests to JSONL file
#             jsonl_path = self.results_dir / f"batch_{batch_num:03d}_requests.jsonl"
#
#             with open(jsonl_path, 'w') as f:
#                 for request in batch['requests']:
#                     f.write(json.dumps(request) + '\n')
#
#             # Upload file to OpenAI
#             with open(jsonl_path, 'rb') as f:
#                 file_response = self.client.files.create(
#                     file=f,
#                     purpose='batch'
#                 )
#
#             # Create batch job
#             batch_response = self.client.batches.create(
#                 input_file_id=file_response.id,
#                 endpoint="/v1/chat/completions",
#                 completion_window="24h"
#             )
#
#             print(f"   ✅ Submitted batch ID: {batch_response.id}")
#             return batch_response.id
#
#         except Exception as e:
#             print(f"   ❌ Error submitting batch: {str(e)}")
#             return None
#
#     def wait_for_batch_completion(self, batch_id: str, batch_num: int,
#                                   progress: Dict, check_interval: int = 300):
#         """
#         Waits for a batch to complete, with periodic status updates
#         check_interval is in seconds (default 5 minutes)
#         """
#         start_time = time.time()
#
#         while True:
#             try:
#                 batch = self.client.batches.retrieve(batch_id)
#
#                 if batch.status == 'completed':
#                     print(f"   ✅ Batch {batch_num} completed!")
#
#                     # Retrieve and save results
#                     self.retrieve_batch_results(batch_id, batch_num)
#
#                     # Update progress
#                     progress['batches'][batch_num]['status'] = 'completed'
#                     progress['batches'][batch_num]['completed_at'] = datetime.now().isoformat()
#                     self.save_progress(progress)
#
#                     return True
#
#                 elif batch.status == 'failed':
#                     print(f"   ❌ Batch {batch_num} failed!")
#                     progress['batches'][batch_num]['status'] = 'failed'
#                     self.save_progress(progress)
#                     return False
#
#                 else:
#                     # Still processing
#                     elapsed = (time.time() - start_time) / 60
#                     if batch.request_counts:
#                         completed = batch.request_counts.completed
#                         total = batch.request_counts.total
#                         print(f"   ⏳ Status: {batch.status} | Progress: {completed}/{total} | "
#                               f"Elapsed: {elapsed:.1f} minutes")
#
#                     time.sleep(check_interval)
#
#             except Exception as e:
#                 print(f"   ⚠️ Error checking status: {str(e)}")
#                 time.sleep(check_interval)
#
#     def retrieve_batch_results(self, batch_id: str, batch_num: int):
#         """
#         Downloads and saves results from a completed batch
#         """
#         try:
#             batch = self.client.batches.retrieve(batch_id)
#
#             if batch.output_file_id:
#                 # Download results
#                 content = self.client.files.content(batch.output_file_id)
#
#                 # Save to file
#                 results_path = self.results_dir / f"batch_{batch_num:03d}_results.jsonl"
#                 results_path.write_bytes(content.read())
#
#                 print(f"   💾 Results saved to {results_path}")
#
#         except Exception as e:
#             print(f"   ⚠️ Error retrieving results: {str(e)}")
#
#     def load_progress(self) -> Dict:
#         """Loads progress tracking from file"""
#         if self.progress_file.exists():
#             with open(self.progress_file, 'r') as f:
#                 return json.load(f)
#         return {'batches': {}, 'started_at': datetime.now().isoformat()}
#
#     def save_progress(self, progress: Dict):
#         """Saves progress tracking to file"""
#         with open(self.progress_file, 'w') as f:
#             json.dump(progress, f, indent=2)
#
#     def is_batch_completed(self, batch_num: int, progress: Dict) -> bool:
#         """Checks if a batch was already completed"""
#         batch_info = progress.get('batches', {}).get(str(batch_num), {})
#         return batch_info.get('status') == 'completed'
#
#     def combine_all_results(self) -> Dict:
#         """
#         Combines results from all completed batches into final analysis
#         This is like assembling puzzle pieces into the complete picture
#         """
#         combined_results = {
#             'project_evolution': [],
#             'inquiry_patterns': [],
#             'knowledge_evolution': []
#         }
#
#         # Read all result files
#         result_files = sorted(self.results_dir.glob("batch_*_results.jsonl"))
#
#         for result_file in result_files:
#             with open(result_file, 'r') as f:
#                 for line in f:
#                     result = json.loads(line)
#
#                     # Categorize by analysis type
#                     custom_id = result.get('custom_id', '')
#
#                     if 'project_evolution' in custom_id:
#                         combined_results['project_evolution'].append(result)
#                     elif 'inquiry_patterns' in custom_id:
#                         combined_results['inquiry_patterns'].append(result)
#                     elif 'knowledge_evolution' in custom_id:
#                         combined_results['knowledge_evolution'].append(result)
#
#         # Save combined results
#         combined_path = self.results_dir / "combined_analysis.json"
#         with open(combined_path, 'w') as f:
#             json.dump(combined_results, f, indent=2)
#
#         print(f"\n📊 Combined results saved to {combined_path}")
#         print(f"   Project Evolution: {len(combined_results['project_evolution'])} analyses")
#         print(f"   Inquiry Patterns: {len(combined_results['inquiry_patterns'])} analyses")
#         print(f"   Knowledge Evolution: {len(combined_results['knowledge_evolution'])} analyses")
#
#         return combined_results

#Version 4

# #!/usr/bin/env python3
# """
# Staged Batch Analyzer - Handles token limits intelligently
# This system processes your chat history in stages, respecting OpenAI's 5M token queue limit
# """
#
# import json
# import time
# import sys
# from pathlib import Path
# from datetime import datetime, timedelta
# from typing import List, Dict, Optional, Tuple
# import tiktoken
# from openai import OpenAI
#
# class StagedBatchAnalyzer:
#     """
#     Manages batch submissions within token limits
#     Think of this as a smart loading dock that knows exactly how much can fit in each truck
#     """
#
#     def __init__(self, config):
#         """Initialize with your existing configuration"""
#         self.config = config
#         self.client = OpenAI(api_key=config.OPENAI_API_KEY)
#         self.encoding = tiktoken.encoding_for_model("gpt-4")
#
#         # Token limit with safety buffer (90% of 5M to avoid edge cases)
#         self.MAX_ENQUEUED_TOKENS = 4_500_000
#
#         # Tracking files for staged processing
#         self.progress_file = Path("batch_progress.json")
#         self.results_dir = Path("staged_results")
#         self.results_dir.mkdir(exist_ok=True)
#
#     def estimate_request_tokens(self, messages: List[Dict], max_tokens: int) -> int:
#         """
#         Accurately estimates tokens for a single request
#         This is like weighing a package before shipping
#         """
#         total_tokens = 0
#
#         # Count tokens in all messages
#         for message in messages:
#             if isinstance(message.get('content'), str):
#                 total_tokens += len(self.encoding.encode(message['content']))
#
#         # Add the maximum possible response tokens
#         total_tokens += max_tokens
#
#         # Add ~10% overhead for JSON structure and system prompts
#         total_tokens = int(total_tokens * 1.1)
#
#         return total_tokens
#
#     #Corrected timestamp treatment
#     def filter_conversations_by_date(self, conversations: List[Dict],
#                                      months_back: int = 12) -> List[Dict]:
#         """
#         Filters conversations to only include those from the past N months
#         This helps you focus on recent, relevant history
#         """
#         cutoff_date = datetime.now() - timedelta(days=months_back * 30)
#
#         filtered_conversations = []
#         for conv in conversations:
#             # Get the create_time value - it's a float (Unix timestamp) in your data
#             create_time = conv.get('create_time', 0)
#
#             try:
#                 if isinstance(create_time, (int, float)):
#                     # Unix timestamp - convert to datetime
#                     conv_date = datetime.fromtimestamp(create_time)
#                 elif isinstance(create_time, str):
#                     # ISO format string - parse it (just in case)
#                     if create_time.endswith('Z'):
#                         create_time = create_time.replace('Z', '+00:00')
#                     conv_date = datetime.fromisoformat(create_time)
#                 else:
#                     # Unknown format - skip this conversation
#                     print(f"  ⚠️  Skipping conversation with unknown date format: {type(create_time)}")
#                     continue
#
#                 if conv_date >= cutoff_date:
#                     filtered_conversations.append(conv)
#
#             except (ValueError, TypeError, AttributeError) as e:
#                 print(f"  ⚠️  Could not parse date for conversation: {create_time} - {str(e)}")
#                 continue
#
#         print(f"Filtered to {len(filtered_conversations)} conversations from the past {months_back} months")
#         print(f"Date range: {cutoff_date.strftime('%Y-%m-%d')} to present")
#
#         return filtered_conversations
#
#     # def filter_conversations_by_date(self, conversations: List[Dict],
#     #                                months_back: int = 12) -> List[Dict]:
#     #     """
#     #     Filters conversations to only include those from the past N months
#     #     This helps you focus on recent, relevant history
#     #     """
#     #     cutoff_date = datetime.now() - timedelta(days=months_back * 30)
#     #
#     #     filtered_conversations = []
#     #     for conv in conversations:
#     #         # Parse the conversation date (adjust this based on your data structure)
#     #         conv_date = datetime.fromisoformat(conv.get('create_time', '').replace('Z', '+00:00'))
#     #
#     #         if conv_date >= cutoff_date:
#     #             filtered_conversations.append(conv)
#     #
#     #     print(f"Filtered to {len(filtered_conversations)} conversations from the past {months_back} months")
#     #     print(f"Date range: {cutoff_date.strftime('%Y-%m-%d')} to present")
#     #
#     #     return filtered_conversations
#
#     def create_staged_batches(self, conversations: List[Dict],
#                             chunk_size: int = 50) -> List[Dict]:
#         """
#         Creates batches that fit within token limits
#         Each batch is like a shipping container with a weight limit
#         """
#         # First, create chunks from conversations
#         chunks = self.create_conversation_chunks(conversations, chunk_size)
#
#         # Now group chunks into batches that fit token limits
#         batches = []
#         current_batch = {
#             'requests': [],
#             'estimated_tokens': 0,
#             'chunk_ids': []
#         }
#
#         for chunk in chunks:
#             # Create the three analysis requests for this chunk
#             requests = self.create_analysis_requests(chunk)
#
#             # Estimate tokens for all three requests
#             chunk_tokens = 0
#             for req in requests:
#                 tokens = self.estimate_request_tokens(
#                     req['body']['messages'],
#                     req['body']['max_tokens']
#                 )
#                 chunk_tokens += tokens
#
#             # Check if adding this chunk would exceed limit
#             if current_batch['estimated_tokens'] + chunk_tokens > self.MAX_ENQUEUED_TOKENS:
#                 # Save current batch and start a new one
#                 if current_batch['requests']:
#                     batches.append(current_batch)
#
#                 current_batch = {
#                     'requests': [],
#                     'estimated_tokens': 0,
#                     'chunk_ids': []
#                 }
#
#             # Add chunk to current batch
#             current_batch['requests'].extend(requests)
#             current_batch['estimated_tokens'] += chunk_tokens
#             current_batch['chunk_ids'].append(chunk['chunk_id'])
#
#         # Don't forget the last batch
#         if current_batch['requests']:
#             batches.append(current_batch)
#
#         print(f"\nCreated {len(batches)} staged batches:")
#         for i, batch in enumerate(batches, 1):
#             print(f"  Batch {i}: {len(batch['requests'])} requests, "
#                   f"~{batch['estimated_tokens']:,} tokens")
#
#         return batches
#
#     def create_conversation_chunks(self, conversations: List[Dict],
#                                   chunk_size: int) -> List[Dict]:
#         """
#         Chunks conversations into manageable pieces
#         This is your existing chunking logic, which we'll preserve
#         """
#         chunks = []
#
#         for i in range(0, len(conversations), chunk_size):
#             chunk = conversations[i:i + chunk_size]
#
#             # Create a chunk identifier
#             chunk_id = f"chunk_{i//chunk_size + 1:04d}"
#
#             chunks.append({
#                 'chunk_id': chunk_id,
#                 'conversations': chunk,
#                 'conversation_indices': list(range(i, min(i + chunk_size, len(conversations))))
#             })
#
#         return chunks
#
#     #Version to account for attribute names
#     def create_analysis_requests(self, chunk: Dict) -> List[Dict]:
#         """
#         Creates the three analysis requests for a chunk
#         This preserves your existing three-pronged analysis approach
#         """
#         requests = []
#
#         # Combine conversations into context
#         context = self.format_conversations_for_analysis(chunk['conversations'])
#
#         # Get the prompts - check for different possible attribute names
#         # Your config might use different names, so we'll check for common variations
#         project_prompt = (getattr(self.config, 'PROJECT_ANALYSIS_PROMPT', None) or
#                           getattr(self.config, 'PROJECT_EVOLUTION_PROMPT', None) or
#                           getattr(self.config, 'SYSTEM_PROMPT_PROJECT', None) or
#                           """Analyze these conversations for project evolution and development patterns.
#                           Identify key projects discussed, their progression over time, and major milestones.""")
#
#         inquiry_prompt = (getattr(self.config, 'INQUIRY_ANALYSIS_PROMPT', None) or
#                           getattr(self.config, 'INQUIRY_PATTERNS_PROMPT', None) or
#                           getattr(self.config, 'SYSTEM_PROMPT_INQUIRY', None) or
#                           """Analyze these conversations for patterns in questions and inquiries.
#                           Identify recurring themes, types of problems solved, and learning patterns.""")
#
#         knowledge_prompt = (getattr(self.config, 'KNOWLEDGE_EVOLUTION_PROMPT', None) or
#                             getattr(self.config, 'KNOWLEDGE_ANALYSIS_PROMPT', None) or
#                             getattr(self.config, 'SYSTEM_PROMPT_KNOWLEDGE', None) or
#                             """Analyze these conversations for knowledge evolution and growth.
#                             Track how understanding deepened over time and identify key learning moments.""")
#
#         # Get max tokens with fallback
#         max_tokens = (getattr(self.config, 'MAX_TOKENS_PER_REQUEST', None) or
#                       getattr(self.config, 'MAX_TOKENS', None) or
#                       getattr(self.config, 'MAX_OUTPUT_TOKENS', None) or
#                       4000)  # Default fallback
#
#         # Your three analysis types
#         analysis_types = [
#             {
#                 'custom_id': f"project_evolution-{chunk['chunk_id']}",
#                 'prompt': project_prompt
#             },
#             {
#                 'custom_id': f"inquiry_patterns-{chunk['chunk_id']}",
#                 'prompt': inquiry_prompt
#             },
#             {
#                 'custom_id': f"knowledge_evolution-{chunk['chunk_id']}",
#                 'prompt': knowledge_prompt
#             }
#         ]
#
#         for analysis in analysis_types:
#             request = {
#                 "custom_id": analysis['custom_id'],
#                 "method": "POST",
#                 "url": "/v1/chat/completions",
#                 "body": {
#                     "model": "gpt-4o-mini",
#                     "messages": [
#                         {"role": "system", "content": analysis['prompt']},
#                         {"role": "user", "content": context}
#                     ],
#                     "max_tokens": max_tokens,
#                     "temperature": 0.7
#                 }
#             }
#             requests.append(request)
#
#         return requests
#
#     # def create_analysis_requests(self, chunk: Dict) -> List[Dict]:
#     #     """
#     #     Creates the three analysis requests for a chunk
#     #     This preserves your existing three-pronged analysis approach
#     #     """
#     #     requests = []
#     #
#     #     # Combine conversations into context
#     #     context = self.format_conversations_for_analysis(chunk['conversations'])
#     #
#     #     # Your three analysis types
#     #     analysis_types = [
#     #         {
#     #             'custom_id': f"project_evolution-{chunk['chunk_id']}",
#     #             'prompt': self.config.PROJECT_ANALYSIS_PROMPT
#     #         },
#     #         {
#     #             'custom_id': f"inquiry_patterns-{chunk['chunk_id']}",
#     #             'prompt': self.config.INQUIRY_ANALYSIS_PROMPT
#     #         },
#     #         {
#     #             'custom_id': f"knowledge_evolution-{chunk['chunk_id']}",
#     #             'prompt': self.config.KNOWLEDGE_EVOLUTION_PROMPT
#     #         }
#     #     ]
#     #
#     #     for analysis in analysis_types:
#     #         request = {
#     #             "custom_id": analysis['custom_id'],
#     #             "method": "POST",
#     #             "url": "/v1/chat/completions",
#     #             "body": {
#     #                 "model": "gpt-4o-mini",
#     #                 "messages": [
#     #                     {"role": "system", "content": analysis['prompt']},
#     #                     {"role": "user", "content": context}
#     #                 ],
#     #                 "max_tokens": self.config.MAX_TOKENS_PER_REQUEST,
#     #                 "temperature": 0.7
#     #             }
#     #         }
#     #         requests.append(request)
#     #
#     #     return requests
#
#     def format_conversations_for_analysis(self, conversations: List[Dict]) -> str:
#         """
#         Formats conversations for analysis
#         This is where you'd apply any preprocessing or formatting
#         """
#         formatted_text = ""
#
#         for conv in conversations:
#             formatted_text += f"\n\n--- Conversation ---\n"
#             formatted_text += f"Date: {conv.get('create_time', 'Unknown')}\n"
#             formatted_text += f"Title: {conv.get('title', 'Untitled')}\n\n"
#
#             # Add the actual conversation content
#             # Adjust this based on your data structure
#             for node_id, node_data in conv.get('mapping', {}).items():
#                 message = node_data.get('message')
#                 if message and message.get('content'):
#                     role = message.get('author', {}).get('role', 'unknown')
#                     content = message.get('content', {}).get('parts', [''])[0]
#                     formatted_text += f"{role.upper()}: {content}\n\n"
#
#         return formatted_text
#
#     def submit_staged_batches(self, conversations: List[Dict]) -> Dict:
#         """
#         Main orchestration function - submits batches in stages
#         This is the conductor that manages the entire process
#         """
#         print("\n🚀 Starting Staged Batch Submission Process")
#         print("=" * 60)
#
#         # Filter to past year if specified
#         filtered_convs = self.filter_conversations_by_date(conversations, months_back=12)
#
#         # Create staged batches
#         batches = self.create_staged_batches(filtered_convs)
#
#         # Load or create progress tracking
#         progress = self.load_progress()
#
#         # Process each batch
#         for batch_num, batch in enumerate(batches, 1):
#             print(f"\n📦 Processing Batch {batch_num} of {len(batches)}")
#             print(f"   Chunks: {', '.join(batch['chunk_ids'])}")
#             print(f"   Estimated tokens: {batch['estimated_tokens']:,}")
#
#             # Check if this batch was already processed
#             if self.is_batch_completed(batch_num, progress):
#                 print(f"   ✅ Already completed, skipping...")
#                 continue
#
#             # Submit the batch
#             batch_id = self.submit_single_batch(batch, batch_num)
#
#             if batch_id:
#                 # Update progress
#                 progress['batches'][batch_num] = {
#                     'batch_id': batch_id,
#                     'status': 'submitted',
#                     'submitted_at': datetime.now().isoformat(),
#                     'chunk_ids': batch['chunk_ids']
#                 }
#                 self.save_progress(progress)
#
#                 # Wait for completion before submitting next batch
#                 print(f"   ⏳ Waiting for batch to complete...")
#                 self.wait_for_batch_completion(batch_id, batch_num, progress)
#             else:
#                 print(f"   ❌ Failed to submit batch {batch_num}")
#                 return {'success': False, 'error': f'Failed at batch {batch_num}'}
#
#         print("\n✨ All batches completed successfully!")
#         return {'success': True, 'progress': progress}
#
#     def submit_single_batch(self, batch: Dict, batch_num: int) -> Optional[str]:
#         """
#         Submits a single batch to OpenAI
#         Returns the batch ID if successful
#         """
#         try:
#             # Write requests to JSONL file
#             jsonl_path = self.results_dir / f"batch_{batch_num:03d}_requests.jsonl"
#
#             with open(jsonl_path, 'w') as f:
#                 for request in batch['requests']:
#                     f.write(json.dumps(request) + '\n')
#
#             # Upload file to OpenAI
#             with open(jsonl_path, 'rb') as f:
#                 file_response = self.client.files.create(
#                     file=f,
#                     purpose='batch'
#                 )
#
#             # Create batch job
#             batch_response = self.client.batches.create(
#                 input_file_id=file_response.id,
#                 endpoint="/v1/chat/completions",
#                 completion_window="24h"
#             )
#
#             print(f"   ✅ Submitted batch ID: {batch_response.id}")
#             return batch_response.id
#
#         except Exception as e:
#             print(f"   ❌ Error submitting batch: {str(e)}")
#             return None
#
#     def wait_for_batch_completion(self, batch_id: str, batch_num: int,
#                                  progress: Dict, check_interval: int = 300):
#         """
#         Waits for a batch to complete, with periodic status updates
#         check_interval is in seconds (default 5 minutes)
#         """
#         start_time = time.time()
#
#         while True:
#             try:
#                 batch = self.client.batches.retrieve(batch_id)
#
#                 if batch.status == 'completed':
#                     print(f"   ✅ Batch {batch_num} completed!")
#
#                     # Retrieve and save results
#                     self.retrieve_batch_results(batch_id, batch_num)
#
#                     # Update progress
#                     progress['batches'][batch_num]['status'] = 'completed'
#                     progress['batches'][batch_num]['completed_at'] = datetime.now().isoformat()
#                     self.save_progress(progress)
#
#                     return True
#
#                 elif batch.status == 'failed':
#                     print(f"   ❌ Batch {batch_num} failed!")
#                     progress['batches'][batch_num]['status'] = 'failed'
#                     self.save_progress(progress)
#                     return False
#
#                 else:
#                     # Still processing
#                     elapsed = (time.time() - start_time) / 60
#                     if batch.request_counts:
#                         completed = batch.request_counts.completed
#                         total = batch.request_counts.total
#                         print(f"   ⏳ Status: {batch.status} | Progress: {completed}/{total} | "
#                               f"Elapsed: {elapsed:.1f} minutes")
#
#                     time.sleep(check_interval)
#
#             except Exception as e:
#                 print(f"   ⚠️ Error checking status: {str(e)}")
#                 time.sleep(check_interval)
#
#     def retrieve_batch_results(self, batch_id: str, batch_num: int):
#         """
#         Downloads and saves results from a completed batch
#         """
#         try:
#             batch = self.client.batches.retrieve(batch_id)
#
#             if batch.output_file_id:
#                 # Download results
#                 content = self.client.files.content(batch.output_file_id)
#
#                 # Save to file
#                 results_path = self.results_dir / f"batch_{batch_num:03d}_results.jsonl"
#                 results_path.write_bytes(content.read())
#
#                 print(f"   💾 Results saved to {results_path}")
#
#         except Exception as e:
#             print(f"   ⚠️ Error retrieving results: {str(e)}")
#
#     def load_progress(self) -> Dict:
#         """Loads progress tracking from file"""
#         if self.progress_file.exists():
#             with open(self.progress_file, 'r') as f:
#                 return json.load(f)
#         return {'batches': {}, 'started_at': datetime.now().isoformat()}
#
#     def save_progress(self, progress: Dict):
#         """Saves progress tracking to file"""
#         with open(self.progress_file, 'w') as f:
#             json.dump(progress, f, indent=2)
#
#     def is_batch_completed(self, batch_num: int, progress: Dict) -> bool:
#         """Checks if a batch was already completed"""
#         batch_info = progress.get('batches', {}).get(str(batch_num), {})
#         return batch_info.get('status') == 'completed'
#
#     def combine_all_results(self) -> Dict:
#         """
#         Combines results from all completed batches into final analysis
#         This is like assembling puzzle pieces into the complete picture
#         """
#         combined_results = {
#             'project_evolution': [],
#             'inquiry_patterns': [],
#             'knowledge_evolution': []
#         }
#
#         # Read all result files
#         result_files = sorted(self.results_dir.glob("batch_*_results.jsonl"))
#
#         for result_file in result_files:
#             with open(result_file, 'r') as f:
#                 for line in f:
#                     result = json.loads(line)
#
#                     # Categorize by analysis type
#                     custom_id = result.get('custom_id', '')
#
#                     if 'project_evolution' in custom_id:
#                         combined_results['project_evolution'].append(result)
#                     elif 'inquiry_patterns' in custom_id:
#                         combined_results['inquiry_patterns'].append(result)
#                     elif 'knowledge_evolution' in custom_id:
#                         combined_results['knowledge_evolution'].append(result)
#
#         # Save combined results
#         combined_path = self.results_dir / "combined_analysis.json"
#         with open(combined_path, 'w') as f:
#             json.dump(combined_results, f, indent=2)
#
#         print(f"\n📊 Combined results saved to {combined_path}")
#         print(f"   Project Evolution: {len(combined_results['project_evolution'])} analyses")
#         print(f"   Inquiry Patterns: {len(combined_results['inquiry_patterns'])} analyses")
#         print(f"   Knowledge Evolution: {len(combined_results['knowledge_evolution'])} analyses")
#
#         return combined_results