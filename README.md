# AdaMerging

A code repository of papers about **'[AdaMerging: Adaptive Model Merging for Multi-Task Learning](https://arxiv.org/abs/)'**.



## Abstract
> Multi-task learning (MTL) aims to empower a model to tackle multiple tasks simultaneously. A recent development known as task arithmetic has revealed that several models, each fine-tuned for distinct tasks, can be directly merged into a single model to execute MTL without necessitating a retraining process using the initial training data. Nevertheless, this direct addition of models often leads to a significant deterioration in the overall performance of the merged model. This decline occurs due to potential conflicts and intricate correlations among the multiple tasks. Consequently, the challenge emerges of how to merge pre-trained models more effectively without using their original training data. This paper introduces an innovative technique called Adaptive Model Merging (AdaMerging). This approach aims to autonomously learn the coefficients for model merging, either in a task-wise or layer-wise manner, without relying on the original training data. Specifically, our AdaMerging method operates as an automatic, unsupervised task arithmetic scheme. It leverages entropy minimization on unlabeled test samples from the multi-task setup as a surrogate objective function to iteratively refine the merging coefficients of the multiple models. Our experimental findings across eight tasks demonstrate the efficacy of the AdaMerging scheme we put forth. Compared to the current state-of-the-art (SOTA) task arithmetic merging scheme, AdaMerging showcases a remarkable 11\% improvement in performance. Notably, AdaMerging also exhibits superior generalization capabilities when applied to unseen downstream tasks. Furthermore, it displays a significantly enhanced robustness to data distribution shifts that may occur during the testing phase.

## Citation
If you find our paper or this resource helpful, please consider cite:
```
@article{AdaMerging_Arxiv_2023,
  title={AdaMerging: Adaptive Model Merging for Multi-Task Learning},
  author={Enneng Yang and Zhenyi Wang and Li Shen and Shiwei Liu and Guibing Guo and Xingwei Wang and Dacheng Tao},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```
Thanks!


## Dataset and Checkpoint


## Code