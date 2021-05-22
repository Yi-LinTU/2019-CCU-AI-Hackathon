2019-CCU-AI-Hackathon
------------------------------
## 1. 訓練

1. 準備訓練、驗證集資料
2. 將影像放在一資料夾中 (e.g. train/)
3. 每類影像分別放在相對應類別的資料夾
4. 路徑應如下 :
    ```
    train/
    ├──── 0/
    │     ├──── 0.png
    │     ├──── 1.png
    │     ├──── 2.png
    │     ...
    ├──── 1/
          ├──── 0.png
          ├──── 1.png
          ├──── 2.png
          ...

5. 參數
    * `-c` `--class_num` : number of classes.
    * `-e` `--epochs` : training epochs.
    * `-i` `--img_dir` : training images folder.


6. 執行 `python train.py -c 10 -e 15 -i train`


## 2. Inference
執行 `python test.py -c 10 -i test -w weights/model-1.00-best_train_acc.pth`

