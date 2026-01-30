#!/usr/bin/env python3
"""
Abraham Lincoln LLM Project - Complete Chat Interface
Combines RAG and fine-tuned LLM chat in one interface
"""

import os
import sys
import json
import time
import logging
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LincolnChatOrchestrator:
    """Complete chat interface with both RAG and fine-tuned LLM"""
    
    def __init__(self):
        self.project_root = Path(".")
        self.setup_directories()
        
        # Initialize LLM components
        self.rag_system = None
        
    def setup_directories(self):
        """Setup project directories"""
        self.dirs = {
            'rag_results': self.project_root / 'llm_integration' / 'outputs' / 'rag_results',
            'lora_adapter': self.project_root / 'llm_integration' / 'qwen_0_5b_lora',
            'enhanced_data': self.project_root / 'data_processing' / 'outputs' / 'enhanced_data'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def check_rag_available(self):
        """Check if RAG system is ready"""
        faiss_index = self.dirs['rag_results'] / 'faiss_index.bin'
        metadata_file = self.dirs['rag_results'] / 'documents_metadata.json'
        
        if faiss_index.exists() and metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                if metadata.get('total_documents', 0) > 0:
                    return True, f"✅ RAG ready ({metadata['total_documents']} documents indexed)"
            except:
                pass
        
        return False, "❌ RAG not set up. Documents not indexed."
    
    def check_lora_available(self):
        """Check if LoRA model is ready"""
        lora_dir = self.dirs['lora_adapter']
        
        if lora_dir.exists():
            # Check for adapter files
            adapter_files = list(lora_dir.glob('*.bin'))
            config_files = list(lora_dir.glob('*.json'))
            
            if adapter_files and config_files:
                return True, f"✅ LoRA ready ({len(adapter_files)} adapter files)"
            
            # Check if chat_lora.py script exists
            chat_script = self.project_root / 'llm_integration' / 'chat_lora.py'
            if chat_script.exists():
                return True, f"✅ LoRA chat script available"
        
        return False, "❌ LoRA not fine-tuned. Model not available."
    
    def load_rag_system(self):
        """Load RAG system if not already loaded"""
        if self.rag_system is not None:
            return self.rag_system
        
        try:
            sys.path.insert(0, str(self.project_root / 'llm_integration'))
            from rag_pipeline import LincolnRAGSystem
            
            self.rag_system = LincolnRAGSystem()
            
            # Load index
            if self.rag_system._load_index(str(self.dirs['rag_results'])):
                print("✅ RAG system loaded successfully!")
                return self.rag_system
            else:
                print("❌ Could not load RAG index.")
                return None
                
        except ImportError as e:
            print(f"❌ Error importing RAG: {e}")
            print("\nRequired packages:")
            print("pip install sentence-transformers faiss-cpu")
            return None
        except Exception as e:
            print(f"❌ Error loading RAG: {e}")
            return None
    
    def run_rag_chat(self):
        """Run RAG-based interactive chat"""
        print("\n" + "="*80)
        print("🔍 RAG-BASED LINCOLN CHAT")
        print("="*80)
        print("\nThis system searches through Lincoln's actual documents to answer your questions.")
        print("Commands: 'quit' to exit, 'docs' to see sources, 'help' for more")
        print("-" * 80)
        
        rag_system = self.load_rag_system()
        if rag_system is None:
            print("\n❌ Cannot start RAG chat. Please set up RAG system first.")
            print("Run: python main.py setup-rag")
            return False
        
        print("\n💡 Example questions:")
        print("  • What was Lincoln's view on slavery?")
        print("  • How did Lincoln describe the Union?")
        print("  • Write in the tone of an 1860s presidential proclamation")
        print("-" * 80)
        
        rag_system.interactive_query_mode()
        return True
    
    def run_lora_chat(self):
        """Run LoRA fine-tuned model chat using existing chat_lora.py"""
        print("\n" + "="*80)
        print("🤖 FINE-TUNED LINCOLN MODEL CHAT")
        print("="*80)
        print("\nThis system uses a Qwen1.5-0.5B model fine-tuned on Lincoln's writings.")
        print("It will respond in Abraham Lincoln's 19th-century presidential style.")
        print("Type 'exit' or 'quit' to return to main menu")
        print("-" * 80)
        
        # Check if chat_lora.py exists
        chat_script = self.project_root / 'llm_integration' / 'chat_lora.py'
        
        if not chat_script.exists():
            print(f"❌ Chat script not found: {chat_script}")
            print("\nPlease make sure chat_lora.py exists in llm_integration folder.")
            return False
        
        print("\n🤖 Starting fine-tuned model chat interface...")
        print("💡 Example prompts:")
        print("  • What are your thoughts on democracy?")
        print("  • How should we preserve the Union?")
        print("  • Write a short speech about liberty")
        print("-" * 80)
        
        try:
            # Run the existing chat_lora.py script
            result = subprocess.run(
                [sys.executable, str(chat_script)],
                capture_output=False,  # Don't capture, let it interact directly
                text=False
            )
            
            # Check exit code
            if result.returncode in [0, 130, 2]:  # 0=success, 130=Ctrl+C, 2=quit command
                print("\n✅ LoRA chat session ended.")
                return True
            else:
                print(f"\n❌ LoRA chat exited with code {result.returncode}")
                return False
                
        except FileNotFoundError:
            print(f"❌ Python not found or script missing.")
            return False
        except Exception as e:
            print(f"❌ Error running chat_lora.py: {e}")
            return False
    
    def run_lora_chat_integrated(self):
        """Alternative: Integrated LoRA chat (if you want it built-in)"""
        print("\n" + "="*80)
        print("🤖 FINE-TUNED LINCOLN MODEL CHAT (Integrated)")
        print("="*80)
        
        try:
            # Import from chat_lora.py
            sys.path.insert(0, str(self.project_root / 'llm_integration'))
            
            # We need to run the main function from chat_lora.py
            import importlib.util
            
            spec = importlib.util.spec_from_file_location(
                "chat_lora", 
                str(self.project_root / 'llm_integration' / 'chat_lora.py')
            )
            chat_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(chat_module)
            
            # Run the main function
            chat_module.main()
            return True
            
        except Exception as e:
            print(f"❌ Error running integrated chat: {e}")
            print("\nTrying external script instead...")
            return self.run_lora_chat()
    
    def setup_rag_system(self):
        """Set up the RAG system"""
        print("\n" + "="*80)
        print("🔧 SETTING UP RAG SYSTEM")
        print("="*80)
        
        # Check if enhanced data exists
        enhanced_files = list(self.dirs['enhanced_data'].glob('*.json'))
        if len(enhanced_files) < 10:
            print("❌ Not enough enhanced data files found.")
            print(f"   Found: {len(enhanced_files)} files in {self.dirs['enhanced_data']}")
            print("\nPlease run data processing steps first:")
            print("1. Data cleaning: python data_processing/cleaner.py")
            print("2. Metadata enhancement: python data_processing/metadata_extractor.py")
            return False
        
        print(f"✅ Found {len(enhanced_files)} enhanced documents")
        
        try:
            sys.path.insert(0, str(self.project_root / 'llm_integration'))
            from rag_pipeline import LincolnRAGSystem
            
            rag_system = LincolnRAGSystem()
            
            print("\n📊 Indexing options:")
            print("1. Quick test (100 documents)")
            print("2. Full index (all documents)")
            print("3. Custom sample size")
            
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == '1':
                sample_size = 100
            elif choice == '2':
                sample_size = None  # All documents
            elif choice == '3':
                try:
                    sample_size = int(input("Enter sample size: "))
                except:
                    sample_size = 200
            else:
                sample_size = 100
            
            print(f"\nIndexing {sample_size if sample_size else 'all'} documents...")
            
            result = rag_system.index_documents(
                documents_dir=str(self.dirs['enhanced_data']),
                output_dir=str(self.dirs['rag_results']),
                sample_size=sample_size
            )
            
            if result.get('success', False):
                print(f"✅ RAG system set up successfully!")
                print(f"   Indexed {result['documents_indexed']} documents")
                print(f"   Index saved to: {self.dirs['rag_results']}")
                return True
            else:
                print(f"❌ Failed to set up RAG system: {result.get('error', 'Unknown error')}")
                return False
                
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("\nRequired packages:")
            print("pip install sentence-transformers faiss-cpu")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def setup_lora_fine_tuning(self):
        """Set up LoRA fine-tuning"""
        print("\n" + "="*80)
        print("🔧 SETTING UP LoRA FINE-TUNING")
        print("="*80)
        
        # Check if enhanced data exists
        enhanced_files = list(self.dirs['enhanced_data'].glob('*.json'))
        if len(enhanced_files) < 10:
            print("❌ Not enough enhanced data files found.")
            print(f"   Found: {len(enhanced_files)} files")
            return False
        
        print(f"✅ Found {len(enhanced_files)} enhanced documents")
        
        try:
            ft_script = self.project_root / 'llm_integration' / 'fine_tuning.py'
            
            if not ft_script.exists():
                print(f"❌ Fine-tuning script not found: {ft_script}")
                return False
            
            print("\n⚠️  Fine-tuning will take time and requires significant resources.")
            print("   For Qwen1.5-0.5B, expect 1-3 hours on CPU, less on GPU.")
            print("   Make sure you have at least 8GB RAM available.")
            
            confirm = input("\nProceed with fine-tuning? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Fine-tuning cancelled.")
                return False
            
            print("\nStarting fine-tuning...")
            print("="*80)
            
            original_cwd = os.getcwd()
            os.chdir(ft_script.parent)
            
            try:
                result = subprocess.run(
                    [sys.executable, 'fine_tuning.py'],
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                
                if result.returncode == 0:
                    print("\n✅ Fine-tuning completed successfully!")
                    return True
                else:
                    print(f"\n❌ Fine-tuning failed with exit code {result.returncode}")
                    return False
                    
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def show_main_menu(self):
        """Show main menu with both chat options"""
        print("\n" + "="*80)
        print("🏛️  ABRAHAM LINCOLN LLM - MAIN MENU")
        print("="*80)
        
        # Check what's available
        rag_ready, rag_msg = self.check_rag_available()
        lora_ready, lora_msg = self.check_lora_available()
        
        print("\n🎯 CHAT OPTIONS:")
        print("-" * 80)
        
        if rag_ready:
            print("1. 🔍 RAG-Based Chat")
            print(f"   {rag_msg}")
            print("   • Searches through actual Lincoln documents")
            print("   • Provides factual, document-based answers")
        else:
            print("1. 🔍 RAG-Based Chat (Not set up)")
            print(f"   {rag_msg}")
        
        print()
        
        if lora_ready:
            print("2. 🤖 Fine-Tuned LLM Chat")
            print(f"   {lora_msg}")
            print("   • Uses AI model fine-tuned on Lincoln's writings")
            print("   • Generates creative responses in Lincoln's style")
        else:
            print("2. 🤖 Fine-Tuned LLM Chat (Not set up)")
            print(f"   {lora_msg}")
        
        print("\n3. ⚡ Setup RAG System")
        print("4. ⚙️  Setup Fine-Tuning")
        print("\n0. ↩️  Exit")
        print("-" * 80)
        
        while True:
            choice = input("\nSelect option (0-4): ").strip()
            
            if choice == '0':
                print("\n" + "="*80)
                print("Thank you for using the Abraham Lincoln LLM Project!")
                print("="*80)
                sys.exit(0)
            
            elif choice == '1':
                if rag_ready:
                    self.run_rag_chat()
                    # After chat ends, show menu again
                    print("\n" + "="*80)
                    input("Press Enter to return to main menu...")
                    return True
                else:
                    print("\n❌ RAG system not set up.")
                    print("Please select option 3 to set up RAG system first.")
            
            elif choice == '2':
                if lora_ready:
                    # Try integrated chat first, fall back to script
                    if hasattr(self, 'run_lora_chat_integrated'):
                        self.run_lora_chat_integrated()
                    else:
                        self.run_lora_chat()
                    
                    # After chat ends, show menu again
                    print("\n" + "="*80)
                    input("Press Enter to return to main menu...")
                    return True
                else:
                    print("\n❌ Fine-tuned model not available.")
                    print("Please select option 4 to set up fine-tuning first.")
            
            elif choice == '3':
                if self.setup_rag_system():
                    input("\nPress Enter to return to main menu...")
                return True
            
            elif choice == '4':
                if self.setup_lora_fine_tuning():
                    input("\nPress Enter to return to main menu...")
                return True
            
            else:
                print("❌ Invalid choice. Please select 0-4.")
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*80)
        print("ABRAHAM LINCOLN LLM PROJECT - HELP")
        print("="*80)
        
        print("\n📚 COMMANDS:")
        print("-" * 80)
        print("python main.py          Start interactive chat interface")
        print("python main.py setup-rag     Set up RAG system")
        print("python main.py setup-lora    Set up LoRA fine-tuning")
        print("python main.py help          Show this help")
        
        print("\n🔍 RAG SYSTEM (Option 1):")
        print("-" * 80)
        print("• Searches through actual Lincoln documents")
        print("• Provides factual, document-based answers")
        print("• Shows source documents for verification")
        print("• Best for historical accuracy and research")
        
        print("\n🤖 FINE-TUNED LLM (Option 2):")
        print("-" * 80)
        print("• Qwen1.5-0.5B model fine-tuned on Lincoln's writings")
        print("• Generates creative responses in Lincoln's style")
        print("• Better for conversational and creative tasks")
        print("• Responds in 19th-century presidential tone")
        
        print("\n⚡ SETUP REQUIREMENTS:")
        print("-" * 80)
        print("1. Enhanced data in: data_processing/outputs/enhanced_data/")
        print("2. For RAG: pip install sentence-transformers faiss-cpu")
        print("3. For fine-tuning: pip install transformers peft torch")
        
        print("\n📁 PROJECT STRUCTURE:")
        print("-" * 80)
        print("• RAG Index: llm_integration/outputs/rag_results/")
        print("• LoRA Adapter: llm_integration/qwen_0_5b_lora/")
        print("• Chat Script: llm_integration/chat_lora.py")
        
        print("\n" + "="*80)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Abraham Lincoln LLM Project - Complete Chat Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py          # Start chat interface (choose RAG or fine-tuned)
  python main.py setup-rag     # Set up RAG system
  python main.py setup-lora    # Set up LoRA fine-tuning
  python main.py help          # Show help
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        default='chat',
        choices=['chat', 'setup-rag', 'setup-lora', 'help'],
        help='Command to execute (default: chat)'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = LincolnChatOrchestrator()
    
    # Execute command
    if args.command == 'chat':
        while True:
            orchestrator.show_main_menu()
    
    elif args.command == 'setup-rag':
        orchestrator.setup_rag_system()
    
    elif args.command == 'setup-lora':
        orchestrator.setup_lora_fine_tuning()
    
    elif args.command == 'help':
        orchestrator.show_help()
    
    else:
        print(f"Unknown command: {args.command}")
        orchestrator.show_help()

if __name__ == "__main__":
    main()