# BECE

This is the code for the paper "Dispelling the Fake: Social Bot Detection Based on Edge Confidence Evaluation" [pdf](https://wcnux7s65wu6.feishu.cn/file/SxSzbTrgWoKxKsxpje1cRWHlnEd).

Journal Paper [Link](https://ieeexplore.ieee.org/abstract/document/10530431).
## Quick Start

0. Prepare the environment
    ```
    conda create -n bece python=3.9
    conda activate bece
    pip install -r requirements.txt
    ```
1. Download Datasets

    MGTAB:[link](https://drive.usercontent.google.com/download?id=1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe&export=download&authuser=0);  Cresci-15:[link](https://drive.usercontent.google.com/download?id=1AzMUNt70we5G2DShS8hk5qH95VR9HfD3&export=download&authuser=0); Twibot-20:[link](https://github.com/BunsenFeng/TwiBot-20)
2. Pre-process dataset

    ```
    python process_edge.py
    python edge_label.py
    ```
3. Start

    ```
    python model_edge_feature.py  --lambda_2 0.01 --lambda_3 0.01 --log_dir 'model_edge_feature_log.log' --model_file 'model_edge_feature' 
    ```

## Note
Experimental results may vary slightly due to different hardware configurations.

## Citation


@article{qiao2024dispelling,
  
  title={Dispelling the Fake: Social Bot Detection Based on Edge Confidence Evaluation},
  
  author={Qiao, Boyu and Zhou, Wei and Li, Kun and Li, Shilong and Hu, Songlin},
  
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  
  year={2024},
  
  publisher={IEEE}
}
