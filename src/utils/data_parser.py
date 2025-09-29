import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime


class ChatExportParser:
    """
    Fixed parser for JSON chat exports with recursion protection and proper token counting.
    This version handles circular references and ensures token estimation works correctly.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.conversations = []
        # Track visited nodes to prevent infinite recursion
        self.visited_nodes = set()

    def parse_export(self) -> List[Dict[str, Any]]:
        """
        Simplified parsing method that directly processes JSON files.
        Returns a list of standardized conversation dictionaries.
        """

        if not self.file_path.exists():
            raise FileNotFoundError(f"Chat export file not found: {self.file_path}")

        file_extension = self.file_path.suffix.lower()

        print(f"📁 Parsing {file_extension} export file: {self.file_path.name}")

        # Only support JSON files now
        if file_extension != '.json':
            raise ValueError(f"Unsupported file format: {file_extension}. Only JSON files are supported.")

        return self._parse_json_export()

    def _parse_json_export(self) -> List[Dict[str, Any]]:
        """
        Parses JSON chat exports with improved error handling.
        This version catches and reports parsing errors for individual conversations.
        """

        with open(self.file_path, 'r', encoding='utf-8') as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing JSON file: {e}")
                print(f"   File might be corrupted or improperly formatted")
                print(f"   Ensure the JSON file is valid using a JSON validator")
                return []

        conversations = []
        skipped_count = 0

        # Handle different JSON export formats
        if isinstance(raw_data, list):
            conversation_list = raw_data
        elif isinstance(raw_data, dict) and 'conversations' in raw_data:
            conversation_list = raw_data['conversations']
        else:
            conversation_list = [raw_data]

        for i, conversation_data in enumerate(conversation_list):
            try:
                # Reset visited nodes for each conversation to prevent cross-contamination
                self.visited_nodes.clear()
                
                parsed_conversation = self._standardize_conversation(conversation_data, i)
                if parsed_conversation and len(parsed_conversation['messages']) > 1:
                    conversations.append(parsed_conversation)
                elif parsed_conversation and len(parsed_conversation['messages']) <= 1:
                    # Skip conversations with too few messages but don't count as error
                    skipped_count += 1
            except RecursionError:
                print(f"⚠️ Skipping conversation {i} due to circular reference in mapping structure")
                skipped_count += 1
            except Exception as e:
                print(f"⚠️ Skipping conversation {i} due to parsing error: {e}")
                skipped_count += 1

        print(f"✅ Successfully parsed {len(conversations)} conversations from JSON export")
        if skipped_count > 0:
            print(f"   ℹ️ Skipped {skipped_count} conversations (errors or too few messages)")
        
        return conversations

    def _standardize_conversation(self, conversation_data: Dict, conversation_id: int) -> Optional[Dict[str, Any]]:
        """
        Converts raw conversation data into our standardized format.
        Now includes proper token counting for cost estimation.
        """

        standardized_conversation = {
            'id': conversation_data.get('id', f'conv_{conversation_id}'),
            'title': conversation_data.get('title', f'Conversation {conversation_id}'),
            'create_time': conversation_data.get('create_time', conversation_data.get('created_at', '')),
            'messages': [],
            'total_tokens': 0,  # This is crucial for cost estimation
            'metadata': {
                'source_format': 'json',
                'original_keys': list(conversation_data.keys())
            }
        }

        # Extract messages - this handles the nested mapping structure
        messages_data = []
        if 'mapping' in conversation_data:
            # ChatGPT export format with mapping - use recursion protection
            messages_data = self._extract_messages_from_mapping(conversation_data['mapping'])
        elif 'messages' in conversation_data:
            # Direct messages array format
            messages_data = conversation_data['messages']

        # Process each message and accumulate tokens
        for message_data in messages_data:
            processed_message = self._process_message(message_data)
            if processed_message:
                standardized_conversation['messages'].append(processed_message)
                # Accumulate tokens for cost estimation
                token_count = processed_message.get('token_estimate', 0)
                standardized_conversation['total_tokens'] += token_count

        # Also count tokens in the title
        if standardized_conversation['title']:
            standardized_conversation['total_tokens'] += self._estimate_tokens(standardized_conversation['title'])

        return standardized_conversation if standardized_conversation['messages'] else None

    def _extract_messages_from_mapping(self, mapping: Dict) -> List[Dict]:
        """
        Extracts messages from the nested mapping structure with circular reference protection.
        This version prevents infinite recursion by tracking visited nodes.
        """
        messages = []
        
        # Find the root node(s) - nodes without parents or with null parents
        root_nodes = []
        for node_id, node_data in mapping.items():
            if node_data.get('parent') is None:
                root_nodes.append(node_id)
        
        # Traverse from each root to extract messages in order
        for root_id in root_nodes:
            # Use a depth limit as an additional safety measure
            self._traverse_mapping_tree(mapping, root_id, messages, depth=0, max_depth=1000)
        
        return messages

    def _traverse_mapping_tree(self, mapping: Dict, node_id: str, messages: List, depth: int = 0, max_depth: int = 1000):
        """
        Recursively traverses the mapping tree with protection against circular references.
        This fixed version prevents infinite recursion using visited tracking and depth limits.
        """
        
        # Protection against infinite recursion
        if depth > max_depth:
            print(f"   ⚠️ Maximum depth {max_depth} reached in conversation tree")
            return
        
        # Check if we've already visited this node (circular reference protection)
        if node_id in self.visited_nodes:
            return
        
        # Check if node exists in mapping
        if node_id not in mapping:
            return
        
        # Mark this node as visited
        self.visited_nodes.add(node_id)
        
        node = mapping[node_id]
        
        # Add the message if it exists and has content
        if node.get('message'):
            message = node['message']
            # Only add messages with actual content
            if message and (message.get('content') or message.get('text')):
                messages.append(message)
        
        # Traverse children with increased depth
        children = node.get('children', [])
        for child_id in children:
            # Only traverse if we haven't visited this child yet
            if child_id not in self.visited_nodes:
                self._traverse_mapping_tree(mapping, child_id, messages, depth + 1, max_depth)

    def _process_message(self, message_data: Dict) -> Optional[Dict[str, Any]]:
        """
        Processes individual messages into standardized format with proper token counting.
        This version ensures token estimates are always calculated.
        """

        if not message_data:
            return None

        # Extract text content from various formats
        content_text = self._extract_text_content(message_data.get('content', message_data.get('text', '')))
        
        # Skip empty or very short messages
        if not content_text or len(content_text.strip()) < 2:
            return None

        # Determine message role (user, assistant, system)
        author_info = message_data.get('author', {})
        role = author_info.get('role', 'unknown')
        
        # Handle various role formats
        if role not in ['user', 'assistant', 'system']:
            if 'user' in str(author_info).lower():
                role = 'user'
            elif 'assistant' in str(author_info).lower() or 'chatgpt' in str(author_info).lower():
                role = 'assistant'
            else:
                # Default to user for unknown roles
                role = 'user'

        # Calculate token estimate - this is crucial for cost estimation
        token_estimate = self._estimate_tokens(content_text)

        processed_message = {
            'role': role,
            'content': content_text,
            'timestamp': message_data.get('create_time', message_data.get('timestamp', '')),
            'token_estimate': token_estimate,  # This must have a value for cost estimation
            'metadata': {
                'original_author': author_info,
                'message_id': message_data.get('id', ''),
                'content_length': len(content_text)
            }
        }

        return processed_message

    def _extract_text_content(self, content_data) -> str:
        """
        Extracts plain text from various content formats in JSON.
        Enhanced to handle more content structures found in ChatGPT exports.
        """

        if content_data is None:
            return ""

        if isinstance(content_data, str):
            return content_data

        if isinstance(content_data, list):
            text_parts = []
            for part in content_data:
                if isinstance(part, dict):
                    if part.get('type') == 'text' and part.get('text'):
                        text_parts.append(part['text'])
                    elif part.get('type') == 'thoughts' and part.get('text'):
                        # Include thoughts as they contain valuable context
                        text_parts.append(f"[Thought: {part['text']}]")
                    elif part.get('type') == 'code' and part.get('text'):
                        text_parts.append(f"```\n{part['text']}\n```")
                    elif 'text' in part:
                        text_parts.append(str(part['text']))
                elif isinstance(part, str):
                    text_parts.append(part)
            return '\n'.join(text_parts)

        if isinstance(content_data, dict):
            # Handle various dict formats
            if content_data.get('type') == 'text':
                return content_data.get('text', '')
            elif 'text' in content_data:
                return str(content_data['text'])
            elif 'content' in content_data:
                # Recursive call for nested content
                return self._extract_text_content(content_data['content'])
            elif 'parts' in content_data:
                # Handle multi-part content
                return self._extract_text_content(content_data['parts'])
            elif 'message' in content_data:
                # Some formats nest the actual content
                return self._extract_text_content(content_data['message'])

        # Fallback: convert to string
        return str(content_data)

    def _estimate_tokens(self, text: str) -> int:
        """
        Provides token count estimation for cost calculation.
        This is critical for the cost estimation feature to work properly.
        
        Using a more accurate estimation based on OpenAI's typical tokenization:
        - Average English text: ~4 characters per token
        - Code and technical content: ~3 characters per token
        - Very short text: minimum 1 token
        """
        if not text:
            return 0
        
        # Check if this looks like code (has common code indicators)
        is_code = any(indicator in text for indicator in ['```', 'def ', 'function', 'import', 'class ', '();', '{}'])
        
        if is_code:
            # Code typically has more tokens per character
            estimated = max(1, len(text) // 3)
        else:
            # Regular text estimation
            estimated = max(1, len(text) // 4)
        
        return estimated


def load_and_parse_chat_export(file_path: str) -> List[Dict[str, Any]]:
    """
    Convenience function to load and parse chat export data.
    Now includes validation that token counting is working.
    """

    # Validate file extension
    if not file_path.endswith('.json'):
        print(f"⚠️ Warning: Expected JSON file but got: {file_path}")
        print(f"   Converting file extension to .json")
        file_path = file_path.rsplit('.', 1)[0] + '.json'

    parser = ChatExportParser(file_path)
    conversations = parser.parse_export()

    # Validate that token counting is working
    total_tokens = sum(conv.get('total_tokens', 0) for conv in conversations)
    if conversations and total_tokens == 0:
        print("⚠️ Warning: Token counting may not be working properly")
        print("   This will affect cost estimation accuracy")
    else:
        avg_tokens = total_tokens // len(conversations) if conversations else 0
        print(f"📊 Token counting verified: {total_tokens:,} total tokens (~{avg_tokens:,} per conversation)")

    # Ensure the processed directory exists before trying to save
    from config.analysis_config import AnalysisConfig
    processed_dir = AnalysisConfig.DATA_PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save processed data for future reference
    processed_file_path = processed_dir / "parsed_conversations.json"

    try:
        with open(processed_file_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved processed conversations to: {processed_file_path}")

    except Exception as e:
        print(f"⚠️ Could not save processed conversations: {e}")

    return conversations