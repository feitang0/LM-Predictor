# LM-Predictor Scratchpad

**Working Document** - Active issues, debugging notes, and temporary work

## Known Issues & Solutions

### Original Design Problems (RESOLVED)

**1. Function Storage in JSON**
- **Problem**: Storing executable code as strings in JSON is unsafe and unvalidable
- **Solution**: Use formula templates with parameter substitution + generate Python modules
- **Status**: ✅ RESOLVED - Updated JSON schema to use formula templates

**2. Deep Module Dependencies**
- **Problem**: Complex modules depend on other modules, but analysis treats them independently
- **Solution**: Nested formula system with recursive evaluation engine
- **Status**: ✅ RESOLVED - Designed nested formula syntax with `{Module}()` calls

**3. Naming Conflicts**
- **Problem**: Multiple libraries have modules with same names (e.g., `Linear` in torch vs transformers)
- **Solution**: Library-based namespacing with full path preservation
- **Status**: ✅ RESOLVED - Implemented `library__path__module` naming scheme

**4. Static Analysis Limitations**
- **Problem**: Source code analysis can't capture runtime optimizations and conditional paths
- **Note**: Acknowledged limitation - formulas represent theoretical computation
- **Status**: ⚠️ ACCEPTED - This is a fundamental limitation of static analysis

**5. Module Import Conflicts**
- **Problem**: Generated module names conflict with actual library imports
- **Solution**: Separate generated_modules namespace with registry-based resolution
- **Status**: ✅ RESOLVED - Designed separate namespace structure

---

## Current Issues

### 🔧 Active Debugging

**None currently**

### ⏳ Pending Investigation

1. **Path-to-Class Name Conversion Edge Cases**
   - Issue: Very long module paths create unwieldy class names
   - Example: `TransformersModelsLlamaModeingLlamaLlamaAttention` (50+ chars)
   - Potential solution: Collision-aware name shortening algorithm

2. **Module Parameter Extraction**
   - Need to determine how to extract module parameters (hidden_size, num_heads, etc.) from model instances
   - Consider using model.config vs inspecting actual tensors

---

## Experimental Ideas

### 🧪 Formula Template Optimizations

**Template Caching Strategy**
- Consider pre-compiling templates to avoid repeated regex parsing
- Investigate template validation at load time

**Memory Analysis Integration**
- Formula templates could include memory access patterns
- Separate reads/writes/intermediates tracking

### 🧪 Agent Integration Approaches

**Prompt Engineering for Agent**
- Need optimal prompt template for analyzing PyTorch forward functions
- Consider using "thinking" mode for complex module analysis

**Batch Analysis**
- Agent could analyze multiple related modules in one session
- Example: All LLaMA modules (attention, MLP, embeddings) together

---

## Temporary Notes

### Development Environment
- Using uv for dependency management (works well)
- PyTorch v2.8.0 as shallow submodule for source analysis
- Transformers pinned to v4.56.0

### Quick Reminders
- Formula syntax: `${param}` for parameters, `{Module}()` for calls
- Naming scheme: `torch.nn.Linear` → `torch__nn__Linear`
- All generated modules go in `generated_modules/library/` structure

### Questions to Investigate Later
1. How to handle optional parameters in formula templates?
2. Should we support conditional formulas (IF statements)?
3. What's the best way to validate generated formulas?

---

## Completed Experiments

### ✅ Documentation Structure (2024-01-15)
- **Experiment**: Split DESIGN.md into focused documents
- **Result**: Much cleaner organization, better token efficiency
- **Action**: Adopted new structure

### ✅ Naming Convention Analysis (2024-01-15)
- **Experiment**: Compared different naming schemes for module conflicts
- **Result**: Library-prefixed double-underscore scheme works best
- **Action**: Implemented in design spec

---

## Related Documentation

- **[DESIGN.md](DESIGN.md)** - System architecture and technical design decisions
- **[PROGRESS.md](PROGRESS.md)** - Implementation milestones and current progress
- **[CLAUDE.md](CLAUDE.md)** - Environment setup and development commands