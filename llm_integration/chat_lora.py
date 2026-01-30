#!/usr/bin/env python3
"""
chat_lora.py - Lincoln Chat with Fallback to Base Model
Intelligent system that tries fine-tuned model first, falls back to base if needed
"""

import os
import torch
import logging
import sys
import re
import time
from pathlib import Path
from typing import Tuple, Optional, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lincoln_chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# CONFIGURATION
# =====================================================

BASE_MODEL = "Qwen/Qwen1.5-0.5B-Chat"
LORA_PATH = Path(__file__).parent / "qwen_0_5b_lora"

# =====================================================
# LINCOLN PERSONA SETTINGS
# =====================================================

LINCOLN_SYSTEM_PROMPT = """You are Abraham Lincoln, the 16th President of the United States (1861-1865). 
You must respond STRICTLY in character as Lincoln would:

1. Use FIRST PERSON ("I", "my", "me", "we")
2. Reference your historical context (Civil War, slavery, Gettysburg Address)
3. Use 19th-century American language patterns
4. If asked about modern topics, relate them to principles of liberty/democracy
5. NEVER mention being an AI, chatbot, or computer program
6. If truly unfamiliar with a topic, politely decline to answer

Remember: You are Abraham Lincoln speaking from the 1860s."""

# =====================================================
# RESPONSE QUALITY DETECTION
# =====================================================

class ResponseQualityAnalyzer:
    """Analyze if response sounds like Lincoln"""
    
    # Lincoln signature phrases (case-insensitive)
    LINCOLN_SIGNATURES = [
        r'\b(four score|gettysburg|emancipation proclamation)\b',
        r'\b(civil war|union|confederacy|secession)\b', 
        r'\b(house divided|malice toward none|charity for all)\b',
        r'\b(by the people|for the people|government of the people)\b',
        r'\b(slavery|bondage|freedom|liberty)\b',
        r'\b(my fellow citizens|my countrymen)\b',
        r'\b(as president|during my administration|in my presidency)\b'
    ]
    
    # Phrases that indicate base model (AI responses)
    AI_INDICATORS = [
        r'\bas (an |a )?(ai|artificial intelligence|language model|chatbot|computer program)\b',
        r'\bi( am|\'m) (an |a )?(ai|model|program|assistant)\b',
        r'\btrained (on|with) data\b',
        r'\blarge language model\b',
        r'\bchatgpt|openai|gpt\b',
        r'\bi don\'t have (personal|human)\b',
        r'\bi cannot (experience|feel|have)\b'
    ]
    
    @staticmethod
    def analyze_response(response: str, prompt: str = "") -> Dict:
        """Analyze response quality and Lincoln-likeness"""
        response_lower = response.lower()
        
        # Check for Lincoln signatures
        lincoln_score = 0
        for pattern in ResponseQualityAnalyzer.LINCOLN_SIGNATURES:
            if re.search(pattern, response_lower, re.IGNORECASE):
                lincoln_score += 1
        
        # Check for AI indicators
        is_ai_response = False
        for pattern in ResponseQualityAnalyzer.AI_INDICATORS:
            if re.search(pattern, response_lower, re.IGNORECASE):
                is_ai_response = True
                break
        
        # Check first person usage
        first_person = bool(re.search(r'\b(I|my|me|we|our)\b', response, re.IGNORECASE))
        
        # Check if response addresses the prompt
        relevance_score = 0
        if prompt:
            prompt_keywords = set(re.findall(r'\b\w{4,}\b', prompt.lower()))
            response_keywords = set(re.findall(r'\b\w{4,}\b', response_lower))
            if prompt_keywords:
                relevance_score = len(prompt_keywords.intersection(response_keywords)) / len(prompt_keywords)
        
        # Calculate overall quality score (0-100)
        quality_score = (
            (min(lincoln_score, 5) / 5 * 40) +  # Lincoln signatures (40%)
            (100 if first_person else 0) * 0.2 +  # First person (20%)
            (relevance_score * 100 * 0.3) +  # Relevance (30%)
            (100 if not is_ai_response else 0) * 0.1  # Not AI (10%)
        )
        
        return {
            "quality_score": min(100, quality_score),
            "is_ai_response": is_ai_response,
            "first_person": first_person,
            "lincoln_signatures": lincoln_score,
            "relevance": relevance_score,
            "should_use_fallback": is_ai_response or quality_score < 40
        }

# =====================================================
# MODEL LOADING
# =====================================================

class LincolnChatSystem:
    """Main chat system with fallback capability"""
    
    def __init__(self):
        self.lincoln_model = None
        self.base_model = None
        self.tokenizer = None
        self.analyzer = ResponseQualityAnalyzer()
        self.fallback_count = 0
        self.total_queries = 0
        self.lincoln_success_rate = 0
        
    def load_models(self) -> bool:
        """Load both Lincoln fine-tuned and base models"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel, PeftConfig
            
            logger.info("🤖 LOADING CHAT SYSTEM...")
            logger.info("="*60)
            
            # 1. Load tokenizer
            logger.info("📥 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"✅ Tokenizer loaded (vocab: {self.tokenizer.vocab_size:,})")
            
            # 2. Determine device
            if torch.backends.mps.is_available():
                device = "mps"
                torch_dtype = torch.float16
                logger.info("🍎 Using Apple Silicon GPU (MPS)")
            elif torch.cuda.is_available():
                device = "cuda"
                torch_dtype = torch.float16
                logger.info("🖥️ Using NVIDIA GPU")
            else:
                device = "cpu"
                torch_dtype = torch.float32
                logger.info("💻 Using CPU")
            
            # 3. Load base model
            logger.info("📥 Loading base model...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            )
            
            if device == "mps":
                self.base_model = self.base_model.to("mps")
            elif device == "cuda":
                self.base_model = self.base_model.cuda()
            
            self.base_model.eval()
            self.base_model.config.use_cache = False
            
            base_params = sum(p.numel() for p in self.base_model.parameters())
            logger.info(f"✅ Base model loaded ({base_params:,} parameters)")
            
            # 4. Try to load Lincoln LoRA adapter
            logger.info("📥 Attempting to load Lincoln LoRA adapter...")
            if LORA_PATH.exists() and any(LORA_PATH.glob("*")):
                try:
                    # Check what files exist
                    files = list(LORA_PATH.glob("*"))
                    logger.info(f"📁 Found {len(files)} files in LoRA directory")
                    
                    # Try to load Peft model
                    self.lincoln_model = PeftModel.from_pretrained(
                        self.base_model,
                        str(LORA_PATH)
                    )
                    self.lincoln_model.eval()
                    
                    # Count parameters
                    trainable_params = sum(p.numel() for p in self.lincoln_model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in self.lincoln_model.parameters())
                    
                    logger.info("✨ LINCOLN MODEL LOADED SUCCESSFULLY!")
                    logger.info(f"   Trainable (Lincoln) parameters: {trainable_params:,}")
                    logger.info(f"   % Fine-tuned: {(trainable_params/total_params*100):.4f}%")
                    
                    # Test Lincoln model
                    test_result = self._test_lincoln_model()
                    if test_result:
                        logger.info("🧪 Lincoln persona test: PASSED")
                    else:
                        logger.warning("⚠️ Lincoln persona test: WEAK - May need fallback")
                    
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load Lincoln adapter: {e}")
                    logger.warning("⚠️ Will use base model only")
                    self.lincoln_model = None
            else:
                logger.warning("⚠️ LoRA directory not found or empty")
                logger.warning("   Using base model only")
                self.lincoln_model = None
            
            return self.lincoln_model is not None
            
        except Exception as e:
            logger.error(f"❌ Critical error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _test_lincoln_model(self) -> bool:
        """Test if Lincoln model responds appropriately"""
        test_prompts = [
            "Who are you?",
            "What is democracy?",
        ]
        
        for prompt in test_prompts:
            response = self._generate_with_model(self.lincoln_model, prompt)
            analysis = self.analyzer.analyze_response(response, prompt)
            
            if analysis["is_ai_response"] or analysis["quality_score"] < 50:
                logger.warning(f"   Test '{prompt[:30]}...': Score {analysis['quality_score']:.1f}")
                return False
        
        return True
    
    def _generate_with_model(self, model, prompt: str, **kwargs) -> str:
        """Generate response using specified model"""
        messages = [
            {"role": "system", "content": LINCOLN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # Move to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 300),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": True,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1,
            }
            
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"[Error: {str(e)}]"
    
    def _enhance_base_response(self, response: str, prompt: str) -> str:
        """Try to make base response sound more Lincoln-like"""
        # Remove AI disclaimers
        ai_patterns = [
            r'As (an |a )?AI[^,.]*,?',
            r'As (a |an )?language model[^,.]*,?',
            r'I( am|\'m) (an |a )?(AI|artificial intelligence)[^,.]*,?',
        ]
        
        for pattern in ai_patterns:
            response = re.sub(pattern, '', response, flags=re.IGNORECASE)
        
        # Try to add Lincoln context if missing
        analysis = self.analyzer.analyze_response(response, prompt)
        
        if not analysis["first_person"] and len(response.split()) > 10:
            # Try to convert to first person
            response = re.sub(r'^(Abraham Lincoln|Lincoln|He|She|They)', 'I', response, flags=re.IGNORECASE)
            if not response.startswith(('I ', 'I\'', 'My ', 'We ')):
                response = "I would say that " + response[0].lower() + response[1:]
        
        return response.strip()
    
    def chat(self, prompt: str) -> Tuple[str, str, Dict]:
        """
        Main chat function with fallback
        
        Returns:
            tuple: (response, model_used, analysis_info)
        """
        self.total_queries += 1
        
        logger.debug(f"Processing query: {prompt[:50]}...")
        
        # Try Lincoln model first (if available)
        if self.lincoln_model is not None:
            start_time = time.time()
            response = self._generate_with_model(self.lincoln_model, prompt)
            lincoln_time = time.time() - start_time
            
            # Analyze response quality
            analysis = self.analyzer.analyze_response(response, prompt)
            analysis["generation_time"] = lincoln_time
            analysis["model_used"] = "lincoln"
            
            # Check if we should use fallback
            if analysis["should_use_fallback"]:
                logger.info(f"⚠️ Lincoln response quality low ({analysis['quality_score']:.1f}), using fallback...")
                self.fallback_count += 1
                
                # Generate with base model
                start_time = time.time()
                base_response = self._generate_with_model(self.base_model, prompt)
                base_time = time.time() - start_time
                
                # Enhance base response
                enhanced_response = self._enhance_base_response(base_response, prompt)
                
                # Analyze base response
                base_analysis = self.analyzer.analyze_response(enhanced_response, prompt)
                base_analysis["generation_time"] = base_time
                base_analysis["model_used"] = "base"
                base_analysis["was_fallback"] = True
                base_analysis["original_lincoln_score"] = analysis["quality_score"]
                
                # Update statistics
                self.lincoln_success_rate = ((self.total_queries - self.fallback_count) / self.total_queries) * 100
                
                return enhanced_response, "base_with_fallback", base_analysis
            
            # Lincoln model response is good enough
            self.lincoln_success_rate = ((self.total_queries - self.fallback_count) / self.total_queries) * 100
            return response, "lincoln", analysis
        
        else:
            # Only base model available
            logger.info("ℹ️ Using base model (Lincoln model not available)")
            start_time = time.time()
            response = self._generate_with_model(self.base_model, prompt)
            base_time = time.time() - start_time
            
            # Enhance and analyze
            enhanced_response = self._enhance_base_response(response, prompt)
            analysis = self.analyzer.analyze_response(enhanced_response, prompt)
            analysis["generation_time"] = base_time
            analysis["model_used"] = "base_only"
            analysis["was_fallback"] = False
            
            self.fallback_count = self.total_queries  # All queries are fallback
            self.lincoln_success_rate = 0
            
            return enhanced_response, "base_only", analysis
    
    def get_statistics(self) -> Dict:
        """Get usage statistics"""
        return {
            "total_queries": self.total_queries,
            "fallback_count": self.fallback_count,
            "lincoln_success_rate": self.lincoln_success_rate,
            "fallback_rate": (self.fallback_count / self.total_queries * 100) if self.total_queries > 0 else 0,
            "lincoln_model_available": self.lincoln_model is not None
        }

# =====================================================
# USER INTERFACE
# =====================================================

def display_response(response: str, model_used: str, analysis: Dict, show_details: bool = True):
    """Display response with model info"""
    # Model indicator colors
    model_colors = {
        "lincoln": "\033[92m",  # Green
        "base_with_fallback": "\033[93m",  # Yellow
        "base_only": "\033[91m",  # Red
    }
    
    color = model_colors.get(model_used, "\033[0m")
    reset = "\033[0m"
    
    print(f"\n{color}╔{'═'*68}╗{reset}")
    
    # Model indicator
    if model_used == "lincoln":
        model_text = "🎩 PRESIDENT LINCOLN (Fine-tuned Model)"
    elif model_used == "base_with_fallback":
        model_text = "⚠️  BASE MODEL (Lincoln model quality low)"
    else:
        model_text = "🤖 BASE MODEL (Lincoln model not available)"
    
    print(f"{color}║ {model_text:<66} ║{reset}")
    print(f"{color}╚{'═'*68}╝{reset}")
    
    # Display response
    print(f"\n{response}\n")
    
    # Show details if requested
    if show_details:
        print(f"\n📊 Response Analysis:")
        print(f"   Quality Score: {analysis['quality_score']:.1f}/100")
        
        if model_used == "base_with_fallback":
            print(f"   ⚠️  Original Lincoln score: {analysis.get('original_lincoln_score', 'N/A')}/100")
        
        print(f"   First Person: {'✅ Yes' if analysis['first_person'] else '❌ No'}")
        print(f"   Lincoln Phrases: {analysis['lincoln_signatures']}")
        print(f"   AI Indicators: {'❌ Detected' if analysis['is_ai_response'] else '✅ None'}")
        print(f"   Generation Time: {analysis.get('generation_time', 0):.2f}s")
        
        if analysis.get('was_fallback', False):
            print(f"   ⚠️  This response used FALLBACK to base model")

def main():
    """Main interactive chat loop"""
    print("\n" + "="*80)
    print("🏛️  ABRAHAM LINCOLN CHAT WITH INTELLIGENT FALLBACK")
    print("="*80)
    print("\nSystem will:")
    print("  1. Try Lincoln fine-tuned model first")
    print("  2. Fall back to base model if Lincoln response is poor")
    print("  3. Always inform you which model is being used")
    print("="*80)
    
    # Initialize chat system
    chat_system = LincolnChatSystem()
    
    if not chat_system.load_models():
        print("\n❌ Failed to load models. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("💬 CHAT READY - Type your questions")
    print("="*80)
    print("\nCommands:")
    print("  /stats - Show model usage statistics")
    print("  /details on|off - Toggle detailed analysis display")
    print("  /test - Test Lincoln persona")
    print("  /quit - Exit")
    print("="*80)
    
    show_details = True
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("\nFarewell, and may God bless the United States of America.")
                break
            
            elif user_input.lower() == "/stats":
                stats = chat_system.get_statistics()
                print("\n📈 USAGE STATISTICS:")
                print(f"   Total Queries: {stats['total_queries']}")
                print(f"   Fallbacks: {stats['fallback_count']}")
                print(f"   Lincoln Success Rate: {stats['lincoln_success_rate']:.1f}%")
                print(f"   Fallback Rate: {stats['fallback_rate']:.1f}%")
                print(f"   Lincoln Model: {'✅ Available' if stats['lincoln_model_available'] else '❌ Not Available'}")
                continue
            
            elif user_input.lower().startswith("/details "):
                arg = user_input.split()[1].lower()
                if arg in ["on", "yes", "true"]:
                    show_details = True
                    print("✅ Detailed analysis ON")
                elif arg in ["off", "no", "false"]:
                    show_details = False
                    print("✅ Detailed analysis OFF")
                else:
                    print("Usage: /details on|off")
                continue
            
            elif user_input.lower() == "/test":
                print("\n🧪 TESTING LINCOLN PERSONA...")
                test_prompts = [
                    "Who are you?",
                    "What did you say at Gettysburg?",
                    "Tell me about modern computers.",
                ]
                
                for prompt in test_prompts:
                    print(f"\nTest: '{prompt}'")
                    response, model, analysis = chat_system.chat(prompt)
                    
                    # Show brief result
                    if model == "lincoln":
                        status = "✅ Lincoln Model"
                    elif model == "base_with_fallback":
                        status = "⚠️  Fallback to Base"
                    else:
                        status = "🤖 Base Only"
                    
                    print(f"   {status} (Score: {analysis['quality_score']:.1f})")
                    print(f"   Response: {response[:80]}...")
                
                continue
            
            # Process regular query
            print("\n" + "="*80)
            print("🤔 Thinking...", end="", flush=True)
            
            response, model_used, analysis = chat_system.chat(user_input)
            
            # Clear "Thinking..." line
            print("\r" + " " * 50 + "\r", end="")
            
            # Display response
            display_response(response, model_used, analysis, show_details)
            
            # Show fallback warning if applicable
            if model_used == "base_with_fallback":
                print("\n⚠️  NOTE: The Lincoln model response was of low quality,")
                print("   so the system fell back to the base model response.")
                print("   The base response has been enhanced to sound more Lincoln-like.")
            
            print("\n" + "-"*80)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Conversation interrupted.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Final statistics
    print("\n" + "="*80)
    print("📊 FINAL STATISTICS")
    print("="*80)
    stats = chat_system.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.1f}%")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)