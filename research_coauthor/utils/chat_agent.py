"""
Follow-up Chat Agent for PaperPilot
Handles user questions about generated papers and paper modification requests
"""
import re
from typing import Dict, List, Tuple, Any
from .model_config import generate_with_optimal_model, TaskType

class PaperChatAgent:
    """Interactive chat agent for paper follow-up questions and modifications"""
    
    def __init__(self):
        self.conversation_history = []
        self.max_history = 10  # Limit conversation history for token management
    
    def classify_user_intent(self, user_message: str, paper_content: str) -> str:
        """Classify if user wants to ask questions or modify the paper"""
        modification_keywords = [
            'change', 'modify', 'edit', 'update', 'revise', 'rewrite', 'improve',
            'add', 'remove', 'delete', 'replace', 'fix', 'correct', 'adjust'
        ]
        
        question_keywords = [
            'what', 'why', 'how', 'when', 'where', 'explain', 'clarify',
            'summarize', 'tell me', 'describe', 'elaborate'
        ]
        
        user_lower = user_message.lower()
        
        # Check for modification intent
        if any(keyword in user_lower for keyword in modification_keywords):
            return "modification"
        elif any(keyword in user_lower for keyword in question_keywords):
            return "question"
        else:
            # Default to question if unclear
            return "question"
    
    def answer_paper_question(self, user_question: str, paper_content: str, paper_sections: Dict[str, List[str]]) -> str:
        """Answer questions about the paper content"""
        
        # Build context-aware prompt
        prompt = f"""You are an AI assistant helping users understand a research paper. 
Answer the user's question about the paper clearly and concisely.

Paper Content:
{paper_content[:3000]}  # Limit context to manage tokens

Paper Sections Available:
{', '.join(paper_sections.keys())}

User Question: {user_question}

Provide a helpful, accurate answer based on the paper content. If the question is about a specific section, focus on that section. Keep your response under 500 words."""

        try:
            response = generate_with_optimal_model(TaskType.CHAT, prompt, max_output_tokens=1024)
            return response.text or "I couldn't generate a response. Please try rephrasing your question."
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def modify_paper_section(self, modification_request: str, paper_content: str, paper_sections: Dict[str, List[str]]) -> Dict[str, Any]:
        """Modify paper based on user request"""
        
        # Identify which section to modify
        section_to_modify = self._identify_target_section(modification_request, paper_sections)
        
        if section_to_modify:
            section_content = "\n\n".join(paper_sections[section_to_modify])
        else:
            section_content = paper_content[:2000]  # Use full paper if section unclear
        
        # Build modification prompt
        prompt = f"""You are an AI research assistant. The user wants to modify a research paper section.

Current Section: {section_to_modify or 'Full Paper'}
Current Content:
{section_content}

User's Modification Request: {modification_request}

Please provide the modified content that addresses the user's request. 
- Keep the academic tone and style
- Maintain proper citations in {{n}} format if they exist
- Only modify what the user specifically requested
- Ensure the content flows well

Return only the modified content, no explanations."""

        try:
            response = generate_with_optimal_model(TaskType.GENERATION, prompt, max_output_tokens=1536)
            modified_content = response.text or ""
            
            return {
                "success": True,
                "modified_section": section_to_modify,
                "modified_content": modified_content,
                "modification_type": "section" if section_to_modify else "full"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "modified_section": None,
                "modified_content": "",
                "modification_type": None
            }
    
    def _identify_target_section(self, modification_request: str, paper_sections: Dict[str, List[str]]) -> str:
        """Identify which section the user wants to modify"""
        request_lower = modification_request.lower()
        
        section_keywords = {
            "abstract": ["abstract", "summary"],
            "introduction": ["introduction", "intro", "beginning"],
            "literature review": ["literature", "review", "related work", "background"],
            "methodology": ["methodology", "method", "approach", "technique"],
            "experiments / results": ["results", "experiment", "findings", "data", "analysis"],
            "conclusion": ["conclusion", "ending", "summary", "final"]
        }
        
        for section_name, keywords in section_keywords.items():
            if section_name.lower() in paper_sections:
                if any(keyword in request_lower for keyword in keywords):
                    return section_name
        
        # If no specific section found, return None (modify full paper)
        return None
    
    def add_to_history(self, user_message: str, bot_response: str):
        """Add conversation to history with token management"""
        self.conversation_history.append({
            "user": user_message,
            "bot": bot_response
        })
        
        # Keep only recent conversations to manage tokens
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_context(self) -> str:
        """Get recent conversation history for context"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['bot'][:200]}...")  # Truncate bot responses
        
        return "\n".join(context_parts)
    
    def process_user_input(self, user_message: str, paper_content: str, paper_sections: Dict[str, List[str]]) -> Dict[str, Any]:
        """Main method to process user input and return appropriate response"""
        
        intent = self.classify_user_intent(user_message, paper_content)
        
        if intent == "modification":
            result = self.modify_paper_section(user_message, paper_content, paper_sections)
            self.add_to_history(user_message, "Paper modified successfully" if result["success"] else "Modification failed")
            return {
                "type": "modification",
                "result": result
            }
        else:
            answer = self.answer_paper_question(user_message, paper_content, paper_sections)
            self.add_to_history(user_message, answer)
            return {
                "type": "question",
                "answer": answer
            }
