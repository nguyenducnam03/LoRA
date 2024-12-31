# LoRA
- The goal to understand how LoRA works conceptually and practically, and to implement it from scratch using Python. 
- Initial experiments are conducted on the MNIST dataset, specifically fine-tuning a model to recognize the digit 9.

# Code:
- SVD.ipynb: Demo SVD, technique to decompose a matrix.
- lora_scratch.ipynb: A pure Python implementation of LoRA applied to the MNIST dataset for fine-tuning a model to recognize digit 9.

# What is LoRA?
LoRA (Low-Rank Adaptation) is a technique used in deep learning to reduce the number of trainable parameters during fine-tuning. It works by decomposing the weight updates into low-rank matrices. 
Which:
1. Reduces computational cost.
2. Allows fine-tuning of large models with limited resources.

# How It Works

LoRA introduces two low-rank matrices, A and B, to approximate the weight update.
It mean, W = W + $\Delta$ ($\Delta = BA$)

Initialy, with W (dxk), we had N params.  
Then, we created A,B (dxr, and rxk) (Follow paper, we created B with zero matrix, and A with follow gaussian distribution).  
So, we got total N + dxr + rxk params. But we only train on dxr + rxk params. It mean we inference on W + BA, but we backpropagation on BA only (freeze W).
This is the main idea of LoRA.  
My error of misunderstanding is that, I think only use BA for inference, then don't know how to backprobagation with only BA, but we inference with both W+BA, then calculate loss on branch BA, and freeze W.
