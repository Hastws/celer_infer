# Quick Start Guide - CelerInfer Optimization

**Last Updated**: January 27, 2025  
**Project**: CelerInfer - Hybrid C++/Python LLM Inference Framework

---

## ⚡ Quick Commands

### Build & Test (Typical Workflow)
```bash
# 1. Clean old artifacts
python -m python clean --build-only

# 2. Build C++ engine
python -m python build

# 3. Run full validation
python -m python run-validation --model minimind

# 4. Benchmark performance
python -m python benchmark --model minimind --iterations 5
```

### Individual Operations
```bash
# List available models
python -m python list-models

# Dump model weights to JSON
python -m python dump --model minimind

# Verify PyTorch↔C++ consistency
python -m python validate --model minimind

# Debug a specific layer
python -m python debug --model minimind --layer 0

# Show help for any command
python -m python build --help
```

---

## 📚 Documentation Navigation

### Start Here
- **[docs/README.md](docs/README.md)** - Documentation hub with quick links
- **[COMPLETION_REPORT.md](COMPLETION_REPORT.md)** - Full optimization details

### Core Documentation
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Design patterns & architecture
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Detailed project layout
- **[docs/DIRECTORY_GUIDE.md](docs/DIRECTORY_GUIDE.md)** - Quick reference
- **[docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)** - Doc index

### Validation & Testing
- **[docs/CONSISTENCY_REPORT.md](docs/CONSISTENCY_REPORT.md)** - Testing methodology
- **[docs/VALIDATION_REPORT.md](docs/VALIDATION_REPORT.md)** - Test results
- **[docs/OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md)** - This optimization

### Model Info
- **[docs/MODELS.md](docs/MODELS.md)** - Model specifications
- **[models/minimind/config.json](models/minimind/config.json)** - MiniMind configuration

---

## 🎯 Common Tasks

### Clean Everything
```bash
python -m python clean  # Removes build dirs + Python cache
```

### Build Only
```bash
python -m python clean --build-only  # Just remove builds, keep cache
python -m python build                # Compile C++
```

### Validate Model
```bash
python -m python dump --model minimind              # Export weights
python -m python validate --model minimind           # Check consistency
# Or combined:
python -m python run-validation --model minimind     # Full pipeline
```

### Benchmark Inference
```bash
python -m python benchmark --model minimind          # 10 iterations (default)
python -m python benchmark --model minimind --iterations 20  # Custom iterations
```

---

## 📁 Where Things Are

| What | Where |
|------|-------|
| Documentation | `docs/` (10 markdown files) |
| Python Tools | `python/tools/` (4 helpers) |
| CLI Entry Point | `python/__main__.py` |
| C++ Code | `cpp/` |
| Models | `models/minimind/` |
| Config | `models/minimind/config.json` |
| Weights | `models/minimind/minimind.json` (generated) |
| Build Output | `cpp/build/base_line_micro` |

---

## 🔧 Python Tools (python/tools/)

| Tool | Purpose | Command |
|------|---------|---------|
| build_helper.py | Build C++ | `python -m python build` |
| validate_helper.py | Full validation | `python -m python run-validation` |
| clean_helper.py | Cleanup | `python -m python clean` |
| benchmark_helper.py | Performance | `python -m python benchmark` |

---

## ✅ Verification

All CLI commands working:
```
✅ list-models
✅ dump
✅ validate
✅ debug
✅ build
✅ run-validation
✅ clean
✅ benchmark
```

Performance metrics:
- Build: ~2 seconds
- Inference: 4.4-4.8ms
- Throughput: 217+ inf/sec

---

## 🚀 Next Steps

1. **First Time**
   ```bash
   python -m python build              # Compile C++ engine
   python -m python run-validation     # Verify everything works
   ```

2. **Development**
   - Modify code in python/ or cpp/
   - Run validation to check consistency
   - Use benchmark to measure improvements

3. **Adding New Model**
   - Follow [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
   - Create python/core/mymodel_model.py
   - Create C++ version in cpp/src/models/mymodel.cpp
   - Register in python/core/__init__.py

---

## 📞 Troubleshooting

### C++ Build Fails
```bash
cd cpp && cmake . && make  # Manual build
# Or use Python CLI:
python -m python build --build-dir build
```

### Python Import Errors
```bash
python -m python list-models  # Should list "minimind"
# Check python path
echo $PYTHONPATH
```

### Validation Errors
```bash
python -m python clean --build-only    # Clean builds
python -m python build                 # Rebuild
python -m python run-validation        # Re-validate
```

---

## 📊 Project Stats

- **Languages**: C++14, Python 3.12
- **Documentation**: 10 files, 88 KB
- **CLI Commands**: 8 total
- **Python Tools**: 4 modules, 357 lines
- **Test Coverage**: 12 scenarios verified
- **Status**: ✅ Production Ready

---

## 🔗 Key Resources

- **Copilot Instructions**: [.github/copilot-instructions.md](.github/copilot-instructions.md)
- **Root README**: [README.md](README.md)
- **Main Docs Hub**: [docs/README.md](docs/README.md)
- **Full Report**: [COMPLETION_REPORT.md](COMPLETION_REPORT.md)

---

**Last Optimization**: January 27, 2025  
**Status**: Complete ✅ All systems operational
