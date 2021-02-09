# EVNavi-Python
  
ソケット通信で横方向偏差を返すやつです。  
[![](https://img.youtube.com/vi/OHavOUVYoo8/0.jpg)](https://www.youtube.com/watch?v=OHavOUVYoo8)
  
## Preparation
* Python（バージョンは何以上とか不明ですが新しげなやつ）
* OpenCV Contribution ... YOLOの使用に必要です  
　→ すでに普通版がある場合は削除（pip uninstall opencv-python）  
　→ pip install opencv-contrib-python  
 
## Usage
* デフォルトのIPは"127.0.0.1", ポートは"3000"
* メッセージは2種類。"request\n" →→ "(double型の横方向偏差)\n"と"finish\n"。それ以外はエラーを吐く
1. カメラをつける
2. Pythonサーバーを立ち上げる
3. EV側をクライアントとして立ち上げる
