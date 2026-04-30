# 🧠 TinyVision: Compact Vision Models with Minimal Parameters

**TinyVision** is an evolving research project focused on designing **ultra-lightweight image classification models** with minimal parameter counts. The goal is to explore what’s *actually necessary* for fundamental vision tasks by combining **handcrafted feature preprocessing** with highly efficient CNN architectures.

📦 **Prev Release**: [v2.0.0](https://github.com/SaptakBhoumik/TinyVision/releases/tag/v2.0.0) 

🔖 **Zenodo DOI of V2**: [10.5281/zenodo.16467349](https://doi.org/10.5281/zenodo.16467349)  

> ⚠️ V2 release **does not include a paper**, but focuses on the **codebase**, experiment results, and reproducible training scripts. A deeper analysis and formal documentation will come in future updates. 

I am still writting the report for V3 but you can read the draft for V3 report [here](paper/pdf/v3.pdf) (not final yet, will be updated soon).

You can also read the note for my personal thought [here](cifar10_classifier/final/V1/note.txt). It is not polished but a good refernce point 

## V3 report coming very soon

---

## 🟩 Latest Addition

I was planning to release the quantization experiments in V3 but I didnt get time + My exams started. But if I wait for my exams to end then it will be too late.

So I am releasing one of the countless experiments I have done on quantization. Kinda proud of it because it is a new quantization algo I made(Well I think it is new). But I admit it is not polished because I made a few mistakes which I realised later. But I am sharing it now because I think it is still interesting. It is 
quantile based(Simple yet robust). 

The test accuracy diffrence between QAT with my quantization and F32 model is around 5% which sounds bad but I think it is good because it is 3 bit quantization, I even quantized the activation to 3 bit(Actually everything that makes sense to quantize is quantized. So few parts are not quantized because it does not make sense to quantize them but most of it is quantized), every layer is quantized(well the ones that makes sense like CNN,Linear etc but not the ones that does not make sense like BatchNorm, ReLU etc), I even quantized the first and last layer and also the first input. Not to mention
this model is not that deep(Deep models benifit from QAT more) and also it is really compact with less than 60k parameters. So overall I think it is a good result

But I do admit the code is kinda shit because I was not focused on the code when I made it + a lot of things I have done that make 0 sense if you dont have all the files(They will make sense only if I upload all the file) but I still think it is interesting

Importants:-Proper control group study and comparision with existing methods is missing. Also the model on which I did QAT is an old version of the model in the sense, I have made a lot of new models which you can see in `cifar10_classifier/final/v1` directory that are much lighter but this QAT study was done before it when my models were not that effecient

Check out `cifar10_classifier/proto` if you are interested

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

