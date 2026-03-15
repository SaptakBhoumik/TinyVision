# 🧠 TinyVision: Compact Vision Models with Minimal Parameters

**TinyVision** is an evolving research project focused on designing **ultra-lightweight image classification models** with minimal parameter counts. The goal is to explore what’s *actually necessary* for fundamental vision tasks by combining **handcrafted feature preprocessing** with highly efficient CNN architectures.
<!-- 
📦 **Current Release**: [v2.0.0](https://github.com/SaptakBhoumik/TinyVision/releases/tag/v2.0.0) 

🔖 **Zenodo DOI**: [10.5281/zenodo.16467349](https://doi.org/10.5281/zenodo.16467349)  

📁 **Latest Results & Code**: See the `cat_vs_dog_classifier/final/v2` directory

> ⚠️ This release **does not include a paper**, but focuses on the **codebase**, experiment results, and reproducible training scripts. A deeper analysis and formal documentation will come in future updates. -->

I am still writting the report for V3 but you can read the draft for V3 report [here](paper/pdf/v3.pdf) (not final yet, will be updated soon).

You can also read the note for my personal thought [here](cifar10_classifier/final/V1/note.txt). It is not polished but a good refernce point 

## V3 report coming very soon

---

## 🚧 Project Status

- ✅ **Cat vs Dog Classification**  
  First completed task using a 25,000-image dataset with filter bank preprocessing + compact CNNs.  
  - Achieved **up to 86.87% test accuracy** with models under **12.5k parameters**
  - Several models under **5k parameters** reached over **83% accuracy**, showcasing strong efficiency-performance trade-offs.
  - 📂 Final results and code for this task are in the `cat_vs_dog_classifier/final/v2` directory.
- ✅ **Cifar10 Classification**  
  Second completed task using the Cifar10 dataset but without the filter bank preprocessing.Just relies on compact CNN architectures.
  - Best results achieved :- 
    - 22.11 k parameters model achieved 87.38% accuracy
    - 31.15 k parameters model achieved 88.43% accuracy
    - And more
  - 📂 Final results and code for this task are in the `cifar10_classifier/final/v1` directory.
---

## 🧪 What's Coming Next
- COMING SOON: **Quantization** experiments
- Add thorough **performance analysis** of model architectures to understand why something works while others don't
- Explore new **vision tasks** (edge detection, object detection, etc.) with compact models
- Expand **documentation**, architecture diagrams, and visualizations
- Log and reflect on **failed or inconclusive experiments** critical for understanding design boundaries

---

## 🤝 Contributing

This project is currently personal and tracks my ongoing experiments.  
I’m **not accepting pull requests**, but you're welcome to:

- 📬 Open an [issue](https://github.com/SaptakBhoumik/TinyVision/issues) for discussion or feedback  
- 💌 Reach me at: `saptakbhoumik.acad@gmail.com`
- 📢 Follow me on [X](https://x.com/saptakbhoumik)

---

## 💡 Philosophy

> Small models aren't just about speed—they’re a design challenge.  
> *How much can we cut before it breaks? What’s essential? What’s fluff?*

TinyVision is my attempt to find those answers.

---

