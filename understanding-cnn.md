---
layout: page
permalink: /understanding-cnn/
---

<a name='vis'></a>

(this page is currently in draft form)

## Визуализация того, чему обучается сверточная нейронная сеть

Несколько способов интерпретации и визуализации сверточных нейронных сетей были опубликованы в последнее время как ответ на распространенную критику о том, что нейронные сети с трудом поддаются интерпретации. В данной статье мы расскажем о некоторых из подходов к визуализации и дадим ссылки на другие статьи по данной тематике.

### Визуализация значений активационной функции и весов первого слоя

**Активации на различных слоях**. Самый простой способ визуализировать нейросеть - это показать активации в процессе прямого прохода. Для ReLU сетей, активации обычно сначала выглядят плотно и кучковато, но при дальнейшем обучении они становятся более разреженными и локализованными. С помощью визуализации можно заметить один из опасных подводных камней функции активации ReLU - некоторые области после активации состоят полностью из нулей, что может указывать на "мертвые" фильтры и констатирует, что скорость обучения (learning rate) слишком велика.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/act1.jpeg" width="49%">
  <img src="/assets/cnnvis/act2.jpeg" width="49%">
  <div class="figcaption">
    Типичная картина активаций на первом свертночном слое (слева) и на 5 сверточном слое (справа) сети AlexNet, которая анализирует изображение кошки. Каждый квадрат показывает карту активаций, соответствующую какому-либо фильтру. Заметьте, что активации разрежены (большинство значений нули, на картинке это соответствует черному цвету) и достаточно локализированы.
  </div>
</div>

**Веса сверточных/полносвязных слоев.** Второй распространенный подход - это визуализация весов нейросети. Наиболее легко интерпретировать веса первого слоя, так как он имеет дело с пикселями изображения напрямую, но также возможно визуализировать и веса более глубоких слоев. Отображение весов полезно потому, что хорошо натренированная сеть обычно имеет четкие и интерпретируемые фильтры без шума. Шумовые фильтры обычно указывают на то, что время обучения нейросети было недостаточно долгим или, возможно, что регуляризация была слишком слабой и это привело к переобучению.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/filt1.jpeg" width="49%">
  <img src="/assets/cnnvis/filt2.jpeg" width="49%">
  <div class="figcaption">
    Типичное изображение фильтров первого сверточного слоя (слева) и второго сверточного слоя (справа) в полностью обученной сети AlexNet. Заметьте, что веса первого уровня выглядят четко и гладко, указывая на то, что обучение нейронной сети сошлось к довольно оптимальному решению. Цветные и черно-белые признаки кластеризованы, потому что AlexNet содержить два отдельных потока обработки изображения и очевидное следствие данной архитектуры, что один поток учится выделять частовстречаемые черно-белые признаки, а второй поток выучивает редковстречающиеся цветные признаки. Веса второго сверточного слоя не совсем интерпретируемы, но очевидно, что изображения являются четкими и гладкими, без шума.
  </div>
</div>

### Retrieving images that maximally activate a neuron

Another visualization technique is to take a large dataset of images, feed them through the network and keep track of which images maximally activate some neuron. We can then visualize the images to get an understanding of what the neuron is looking for in its receptive field. One such visualization (among others) is shown in [Rich feature hierarchies for accurate object detection and semantic segmentation](http://arxiv.org/abs/1311.2524) by Ross Girshick et al.:

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/pool5max.jpeg" width="100%">
  <div class="figcaption">
    Maximally activating images for some POOL5 (5th pool layer) neurons of an AlexNet. The activation values and the receptive field of the particular neuron are shown in white. (In particular, note that the POOL5 neurons are a function of a relatively large portion of the input image!) It can be seen that some neurons are responsive to upper bodies, text, or specular highlights.
  </div>
</div>

One problem with this approach is that ReLU neurons do not necessarily have any semantic meaning by themselves. Rather, it is more appropriate to think of multiple ReLU neurons as the basis vectors of some space that represents in image patches. In other words, the visualization is showing the patches at the edge of the cloud of representations, along the (arbitrary) axes that correspond to the filter weights. This can also be seen by the fact that neurons in a ConvNet operate linearly over the input space, so any arbitrary rotation of that space is a no-op. This point was further argued in [Intriguing properties of neural networks](http://arxiv.org/abs/1312.6199) by Szegedy et al., where they perform a similar visualization along arbitrary directions in the representation space.

### Embedding the codes with t-SNE 

ConvNets can be interpreted as gradually transforming the images into a representation in which the classes are separable by a linear classifier. We can get a rough idea about the topology of this space by embedding images into two dimensions so that their low-dimensional representation has approximately equal distances than their high-dimensional representation. There are many embedding methods that have been developed with the intuition of embedding high-dimensional vectors in a low-dimensional space while preserving the pairwise distances of the points. Among these, [t-SNE](http://lvdmaaten.github.io/tsne/) is one of the best-known methods that consistently produces visually-pleasing results.

To produce an embedding, we can take a set of images and use the ConvNet to extract the CNN codes (e.g. in AlexNet the 4096-dimensional vector right before the classifier, and crucially, including the ReLU non-linearity). We can then plug these into t-SNE and get 2-dimensional vector for each image. The corresponding images can them be visualized in a grid:

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/tsne.jpeg" width="100%">
  <div class="figcaption">
    t-SNE embedding of a set of images based on their CNN codes. Images that are nearby each other are also close in the CNN representation space, which implies that the CNN "sees" them as being very similar. Notice that the similarities are more often class-based and semantic rather than pixel and color-based. For more details on how this visualization was produced the associated code, and more related visualizations at different scales refer to <a href="http://cs.stanford.edu/people/karpathy/cnnembed/">t-SNE visualization of CNN codes</a>.
  </div>
</div>

### Occluding parts of the image

Suppose that a ConvNet classifies an image as a dog. How can we be certain that it's actually picking up on the dog in the image as opposed to some contextual cues from the background or some other miscellaneous object? One way of investigating which part of the image some classification prediction is coming from is by plotting the probability of the class of interest (e.g. dog class) as a function of the position of an occluder object. That is, we iterate over regions of the image, set a patch of the image to be all zero, and look at the probability of the class. We can visualize the probability as a 2-dimensional heat map. This approach has been used in Matthew Zeiler's [Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901):

<div class="fig figcenter fighighlight">
  <img src="/assets/cnnvis/occlude.jpeg" width="100%">
  <div class="figcaption">
    Three input images (top). Notice that the occluder region is shown in grey. As we slide the occluder over the image we record the probability of the correct class and then visualize it as a heatmap (shown below each image). For instance, in the left-most image we see that the probability of Pomeranian plummets when the occluder covers the face of the dog, giving us some level of confidence that the dog's face is primarily responsible for the high classification score. Conversely, zeroing out other parts of the image is seen to have relatively negligible impact.
  </div>
</div>

### Visualizing the data gradient and friends

**Data Gradient**.

[Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](http://arxiv.org/abs/1312.6034)

**DeconvNet**.

[Visualizing and Understanding Convolutional Networks](http://arxiv.org/abs/1311.2901)

**Guided Backpropagation**.

[Striving for Simplicity: The All Convolutional Net](http://arxiv.org/abs/1412.6806)

### Reconstructing original images based on CNN Codes

[Understanding Deep Image Representations by Inverting Them](http://arxiv.org/abs/1412.0035)

### How much spatial information is preserved?

[Do ConvNets Learn Correspondence?](http://papers.nips.cc/paper/5420-do-convnets-learn-correspondence.pdf) (tldr: yes)

### Plotting performance as a function of image attributes

[ImageNet Large Scale Visual Recognition Challenge](http://arxiv.org/abs/1409.0575)

## Fooling ConvNets

[Explaining and Harnessing Adversarial Examples](http://arxiv.org/abs/1412.6572)

## Comparing ConvNets to Human labelers

[What I learned from competing against a ConvNet on ImageNet](http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/)
