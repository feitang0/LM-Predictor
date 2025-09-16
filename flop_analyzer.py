import json
import os
import inspect
from typing import Dict, Any, Optional
from module_analyzer_agent import ModuleAnalyzerAgent


class FlopAnalyzer:
    """FLOP and memory analysis for model modules with Claude Code agent integration."""

    def __init__(self, db_path: str = "module_db.json", use_agent: bool = True):
        self.db_path = db_path
        self.module_db = self.load_database()
        self.use_agent = use_agent
        self.agent = ModuleAnalyzerAgent() if use_agent else None
        self.pending_validation = set()  # Track modules that need human validation
    
    def load_database(self) -> Dict[str, Any]:
        """Load module analysis database."""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {
            "version": "1.0",
            "description": "Module FLOP and memory analysis database",
            "modules": {}
        }

    def save_database(self) -> None:
        """Save module analysis database to disk."""
        with open(self.db_path, 'w') as f:
            json.dump(self.module_db, f, indent=2)
        print(f"Database saved to {self.db_path}")

    def get_module_path(self, module) -> str:
        """Extract Python import path from module object."""
        module_class = module.__class__
        return f"{module_class.__module__}"
    
    def analyze_module(self, module_class: str, module_info: dict, batch_size: int = 1, seq_len: int = 2048) -> Dict[str, Any]:
        """Analyze a single module for FLOP count using cache or agent."""
        if module_class in self.module_db["modules"]:
            # Use cached analysis
            return self._use_cached_analysis(module_class, module_info, batch_size, seq_len)
        elif self.use_agent and self.agent:
            # Unknown module - analyze with Claude Code agent
            return self._analyze_with_agent(module_class, module_info, batch_size, seq_len)
        else:
            # No agent available - mark as unknown
            return self._mark_as_unknown(module_class)

    def _use_cached_analysis(self, module_class: str, module_info: dict, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """Use cached analysis from database."""
        analysis = self.module_db["modules"][module_class]
        params = self.extract_module_params(module_info.get("module", ""))

        # Extract formulas from new schema
        flop_analysis = analysis.get("flop_analysis", {})
        memory_analysis = analysis.get("memory_analysis", {})

        # Calculate FLOPs using stored formula
        flops = self.evaluate_flop_formula(flop_analysis.get("formula", "0"), params, batch_size, seq_len)
        memory_reads = self.evaluate_memory_formula(memory_analysis.get("reads_formula", "0"), params, batch_size, seq_len)
        memory_writes = self.evaluate_memory_formula(memory_analysis.get("writes_formula", "0"), params, batch_size, seq_len)

        return {
            "module_class": module_class,
            "flops": flops,
            "memory_reads": memory_reads,
            "memory_writes": memory_writes,
            "status": "cached",
            "validated": analysis.get("validation", {}).get("status") == "validated"
        }

    def _analyze_with_agent(self, module_class: str, module_info: dict, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """Analyze unknown module with Claude Code agent."""
        print(f"Analyzing {module_class} with Claude Code agent...")

        try:
            # Get module import path
            module_path = self.get_module_path_from_info(module_info)

            # Agent analyzes the forward function
            agent_analysis = self.agent.analyze_module_with_agent(module_class, module_path)

            # Store in database
            self.module_db["modules"][module_class] = agent_analysis
            self.save_database()

            # Mark for human validation
            self.pending_validation.add(module_class)

            # Calculate FLOPs using the generated analysis
            params = self.extract_module_params(module_info.get("module", ""))
            flop_analysis = agent_analysis.get("flop_analysis", {})
            memory_analysis = agent_analysis.get("memory_analysis", {})

            flops = self.evaluate_flop_formula(flop_analysis.get("formula", "0"), params, batch_size, seq_len)
            memory_reads = self.evaluate_memory_formula(memory_analysis.get("reads_formula", "0"), params, batch_size, seq_len)
            memory_writes = self.evaluate_memory_formula(memory_analysis.get("writes_formula", "0"), params, batch_size, seq_len)

            return {
                "module_class": module_class,
                "flops": flops,
                "memory_reads": memory_reads,
                "memory_writes": memory_writes,
                "status": "agent_analyzed",
                "validated": False,
                "thinking_process": flop_analysis.get("thinking_process", "")
            }

        except Exception as e:
            print(f"Agent analysis failed for {module_class}: {e}")
            return self._mark_as_unknown(module_class)

    def _mark_as_unknown(self, module_class: str) -> Dict[str, Any]:
        """Mark module as unknown/unanalyzed."""
        return {
            "module_class": module_class,
            "flops": 0,
            "memory_reads": 0,
            "memory_writes": 0,
            "status": "unknown",
            "validated": False
        }

    def get_module_path_from_info(self, module_info: dict) -> str:
        """Extract module path from module info dict."""
        module_str = module_info.get("module", "")
        # Try to extract class name and guess path
        class_name = module_info.get("class", "")

        # Common module paths
        if "Llama" in class_name:
            return "transformers.models.llama.modeling_llama"
        elif "Linear" in class_name:
            return "torch.nn.modules.linear"
        elif "Embedding" in class_name:
            return "torch.nn.modules.sparse"
        elif "LayerNorm" in class_name:
            return "torch.nn.modules.normalization"
        else:
            # Generic fallback
            return f"torch.nn.modules.{class_name.lower()}"
    
    def extract_module_params(self, module_str: str) -> dict:
        """Extract parameters from module string representation."""
        params = {}
        
        # Common parameter patterns
        param_patterns = [
            "in_features", "out_features", "num_embeddings", "embedding_dim",
            "num_heads", "head_dim", "hidden_size", "intermediate_size"
        ]
        
        for param in param_patterns:
            try:
                start = module_str.find(f"{param}=") + len(f"{param}=")
                if start > len(f"{param}=") - 1:  # Found the parameter
                    end = module_str.find(",", start)
                    if end == -1:
                        end = module_str.find(")", start)
                    params[param] = int(module_str[start:end])
            except:
                pass
        
        # Handle bias parameter
        params["bias"] = "bias=True" in module_str
        
        return params
    
    def evaluate_flop_formula(self, formula: str, params: dict, B: int, S: int) -> int:
        """Evaluate FLOP formula with given parameters."""
        try:
            # Create evaluation context
            context = {"B": B, "S": S, **params}
            # Simple formula evaluation (for basic formulas)
            # Replace common patterns
            formula = formula.replace("in_features", str(params.get("in_features", 4096)))
            formula = formula.replace("out_features", str(params.get("out_features", 4096)))
            formula = formula.replace("num_embeddings", str(params.get("num_embeddings", 32000)))
            formula = formula.replace("embedding_dim", str(params.get("embedding_dim", 4096)))
            formula = formula.replace("B", str(B))
            formula = formula.replace("S", str(S))
            
            # Handle bias condition
            if "if bias" in formula:
                if params.get("bias", False):
                    formula = formula.split(" + (")[1].split(" if bias")[0]
                    return eval(f"2 * {B} * {S} * {params.get('in_features', 4096)} * {params.get('out_features', 4096)} + {formula}")
                else:
                    formula = formula.split(" + (")[0]
                    return eval(formula)
            
            return eval(formula)
        except:
            return 0
    
    def evaluate_memory_formula(self, formula: str, params: dict, B: int, S: int) -> int:
        """Evaluate memory formula with given parameters."""
        try:
            formula = formula.replace("B", str(B))
            formula = formula.replace("S", str(S))
            formula = formula.replace("in_features", str(params.get("in_features", 4096)))
            formula = formula.replace("out_features", str(params.get("out_features", 4096)))
            formula = formula.replace("num_embeddings", str(params.get("num_embeddings", 32000)))
            formula = formula.replace("embedding_dim", str(params.get("embedding_dim", 4096)))
            
            # Handle conditional memory (bias)
            if "if bias" in formula:
                base = formula.split(" + (")[0]
                bias_term = formula.split(" + (")[1].split(" if bias")[0]
                if params.get("bias", False):
                    return eval(base) + eval(bias_term)
                else:
                    return eval(base)
            
            return eval(formula)
        except:
            return 0
    
    def analyze_model(self, model, batch_size: int = 1, seq_len: int = 2048) -> Dict[str, Any]:
        """Analyze PyTorch model for FLOP count using cache and agent."""
        results = {
            "total_flops": 0,
            "total_memory_reads": 0,
            "total_memory_writes": 0,
            "modules": {},
            "unknown_modules": [],
            "agent_analyzed_modules": [],
            "cached_modules": []
        }

        def analyze_module_recursive(module, path: str = ""):
            children = dict(module.named_children())
            module_class = module.__class__.__name__

            if not children:
                # Leaf node - analyze this module
                module_info = {
                    "class": module_class,
                    "module": str(module)
                }
                analysis = self.analyze_module(module_class, module_info, batch_size, seq_len)
                results["modules"][path or module_class] = analysis
                results["total_flops"] += analysis["flops"]
                results["total_memory_reads"] += analysis["memory_reads"]
                results["total_memory_writes"] += analysis["memory_writes"]

                # Track different analysis types
                if analysis["status"] == "unknown":
                    results["unknown_modules"].append(module_class)
                elif analysis["status"] == "agent_analyzed":
                    results["agent_analyzed_modules"].append(module_class)
                elif analysis["status"] == "cached":
                    results["cached_modules"].append(module_class)
            else:
                # Has children - analyze recursively
                for child_name, child_module in children.items():
                    child_path = f"{path}.{child_name}" if path else child_name
                    analyze_module_recursive(child_module, child_path)

        analyze_module_recursive(model)

        # Add validation summary
        results["pending_validation"] = list(self.pending_validation)

        return results
    
    def print_analysis_summary(self, results: dict, batch_size: int, seq_len: int):
        """Print formatted analysis summary."""
        print(f"\n=== FLOP Analysis Summary ===")
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        print(f"Total FLOPs: {results['total_flops']:,} ({results['total_flops']/1e9:.2f}B)")
        print(f"Total Memory Reads: {results['total_memory_reads']:,} bytes ({results['total_memory_reads']/(1024**3):.3f} GB)")
        print(f"Total Memory Writes: {results['total_memory_writes']:,} bytes ({results['total_memory_writes']/(1024**3):.3f} GB)")
        print(f"FLOPs per token: {results['total_flops']/(batch_size * seq_len):,.0f}")

        # Show cached modules
        if results["cached_modules"]:
            cached_unique = list(set(results["cached_modules"]))
            print(f"\n=== Cached Modules ({len(cached_unique)}) ===")
            for module_class in sorted(cached_unique):
                print(f"  ✓ {module_class}")

        # Show agent-analyzed modules
        if results["agent_analyzed_modules"]:
            agent_unique = list(set(results["agent_analyzed_modules"]))
            print(f"\n=== Agent-Analyzed Modules ({len(agent_unique)}) ===")
            for module_class in sorted(agent_unique):
                print(f"  🤖 {module_class}")

        # Show unknown modules
        if results["unknown_modules"]:
            unknown_unique = list(set(results["unknown_modules"]))
            print(f"\n=== Unknown Modules ({len(unknown_unique)}) ===")
            print("These modules could not be analyzed:")
            for module_class in sorted(unknown_unique):
                print(f"  ❓ {module_class}")

        # Show pending validation
        if results.get("pending_validation"):
            print(f"\n=== Pending Human Validation ({len(results['pending_validation'])}) ===")
            print("These agent-analyzed modules need human review:")
            for module_class in sorted(results["pending_validation"]):
                print(f"  ⏳ {module_class}")

        # Show detailed module results
        print(f"\n=== Top FLOP Contributors ===")
        module_flops = [(name, info["flops"]) for name, info in results["modules"].items() if info["flops"] > 0]
        module_flops.sort(key=lambda x: x[1], reverse=True)
        for name, flops in module_flops[:10]:
            percentage = (flops / results["total_flops"]) * 100 if results["total_flops"] > 0 else 0
            print(f"  {name}: {flops:,} FLOPs ({percentage:.1f}%)")