训练语料是从: https://archive.org/details/DeepakChopraHowToKnowGod 网站获得
训练数据保存在./data/train.txt内 是full text中的3个文件合并而来train.txt里面包含很多没用的 "\n"(换行符),前期做了少量的处理
(train.txt最好是能去除一些没用的文字信息)

环境: python 3.6.2
依赖包: pytorch

用LSTM训练一个 language model 
### run this program
1、将训练文本语料放入./data/路径下命名为train.txt
2、修改main-gpu.py内的配置参数(在10+到20行左右)
3、执行 main.gpu.py 如果没有GPU环境建议精简 trian.txt
4、生成的sample保存在./cache/目录下的sample.txt文件内  

### Part B : Wisdom Generator(智慧发生器)

There is a body of text [here](https://archive.org/stream/DeepakChopraHowToKnowGod/Deepak%20Chopra/Deepak%20Chopra%20-%20The%20Seven%20Spiritual%20Laws%20Of%20Success_djvu.txt).

Deepak Chopra is occasionally the subject of [parody](http://www.wisdomofchopra.com/), such as this [New Age Bullshit Generator](http://sebpearce.com/bullshit/) (see also the [blog](http://sebpearce.com/blog/bullshit/) ).
Those parodies are being made from [simple patterns](https://github.com/sebpearce/bullshit/blob/gh-pages/php/patterns.php) and from a [small structured vocabulary](https://github.com/sebpearce/bullshit/blob/gh-pages/vocab.js). Just like [deep drumpf](http://www.csail.mit.edu/deepdrumpf) many of the random creations are hard to tell from the real thing. The pseudo-Chopra efforts use predefined grammatical structures lists of hand-tagged words to achieve realism, while DeepDrumpf uses an RNN and [lots of data](http://www.trumptwitterarchive.com/).

Take the above [text](https://archive.org/stream/DeepakChopraHowToKnowGod/Deepak%20Chopra/Deepak%20Chopra%20-%20The%20Seven%20Spiritual%20Laws%20Of%20Success_djvu.txt).
(and perhaps add all tweets from [here](https://twitter.com/deepakchopra), perhaps using [tweepy](http://www.tweepy.org/). Then there's [this](https://twitter.com/hashtag/cosmisconciousness) too.).

 * try to train an LSTM (or similar RNN) on this data : can you "beat" the simple pattern generator, in some sense? 
 * Describe how you did this, what the results were like, what you tried to "make it work". 
 * Cherry pick some examples of your generated text to illustrate how far it got.

### Part C :
题目大意:
集合T是已知的高质量图片集合，C是在T基础上生成的生成图片集合。P(C|T)是先验概率，记可以理解为T生成C的函数。T^在某种变换下生成C^。现在是不知道T^，如何根据已知的C^、T和P(C|P)求得T^。
使用GAN生成对抗网络完成该题目。
思路是在根据T训练一个生成对抗网络。将P(C|T)作为输出喂给已经训练好的GAN网络中的generator将会得到C(因为有先验概率P(C|T),C的置信度很高)。然后根据C和C^训练一个编码器名叫EDcoder，EDcoder输入输出都是图片，训练数据为C，标签数据为C^。将T输入到训练好的EDcoder网络中可以生成T^。( generator的输入源为高斯噪声)。