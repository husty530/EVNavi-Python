# EVNavi-Python

ソケット通信で横方向偏差を返すやつです。  
[![](https://img.youtube.com/vi/OHavOUVYoo8/0.jpg)](https://www.youtube.com/watch?v=OHavOUVYoo8)

## Usage
* デフォルトのIPは"127.0.0.1", ポートは"3000"
* メッセージは2種類。"request\n" →→ "(double型の横方向偏差)\n"と"finish\n"。それ以外は無視される。
1. カメラをつける
2. C#サーバーを立ち上げる
3. EV側をクライアントとして立ち上げる
