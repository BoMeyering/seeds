---
name: Detect Object Edges and Find Contours
---

## The Canny Algorithm

The Canny edge detection algorithm is commonly employed to detect changes in the intensity of pixels in an image in order to detect the "edge" of an object. It has a few distinct steps that need to happen in order for it to be effective. First, we need to remove noise from the image by a Gaussian filtration step. This has the effect of blurring the source image and making the main features appear more prominent compared to the noise. Next, the image gets filtered with a Sobel kernel to find the gradent directions normal to the edges. The third step involves suppressing the non-maximum intensity pixels. Basically, after finding the gradients, we will be left with a lot of edge lines of varying thickness and pixel intensity. Non-maximum pixels are suppressed by looping over all the pixels in an image, and seeing if the current pixel forms a local maximum with respect to other pixels along the gradient direction. If a given pixel's intensity is not a local maximum along the gradient direction, it is discarded, retained otherwise. The final step is to retain only pixels that are actually edges based on their intensity and connection to other high edge pixels. We employ a hysteresis procedure using high and low threshold values. Briefly, any pixels below the low threshold are discarded since we assume them to be non-edge pixels, any above the high threshold are retained as sure-edge pixels, and any pixels whose intensity lies between the threshold are retained or discarded based on their direct connection to a sure-edge pixel. Not only does this final step generate solid edges that we can use downstream, it also removes any small noise particles that might affect our analysis. You can set lower/higher thresholds to retain/discard more edges. An excellent introduction to the Canny algorithm can be found in this <a href="https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123">TDS article</a> if you'd like to learn more.

## Implementing Canny in OpenCV
In openCV we can use Canny to detect object edges in images. In our image, the background is white while the seeds have fairly distinct boundaries from the background color so we won't have to do any preprocessing on the image.

We will use the function `cv2.Canny()` to return a single channel, binary image where edges are white pixels and non-edge pixels are black. It takes
at minimum three arguments, the `image` source, `threshold1` and `threshold2` for the hysteresis procedure. The code snippet below finds the edges and then prints out the results.
```
>>> edges = cv2.Canny(img, 100, 200)
>>> print(edges)
<class 'numpy.ndarray'>
[[0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 ...
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]
 [0 0 0 ... 0 0 0]]
>>> print(edges.shape)
(3307, 4507)
```

Notice how we dropped the final axis from the image numpy array? Now the edges is just a binary image and we can still view the edges like any other image array.
```
>>> showImage(edges)
```
<img src="img/canny.png" style="max-width: 100%;" title="Seed edges after Canny processing" alt="Seed edges after Canny processing">

 It looks pretty good except you notice that there are some edges that don't quite connect. This can happen because of the thresholds that we set - if we lowered the thresholds, these might disappear. We'll keep the image as is for not though. In addition to these open edgest, there are edges in the middle of the seeds that we don't need. There are a few different ways to tackle this, but I will only demonstrate one here.
 
 We can use morphology operations to close these gaps. Specifically, dilating the image based on a defined kernel size, and then eroding the image again. This helps to remove small gaps in the image. OpenCV has good documentation on the <a href="https://docs.opencv.org/3.4/d3/dbe/tutorial_opening_closing_hats.html" target="_blank" rel="noopener noreferrer">cv2.morphologyEx()</a> on their website. We will use a 5x5 kernel size and implement it as shown below. `cv2.morphologyEx()` takes a `src` image, a morphological operation function `op`, and an `kernel` structuring element. All of the morphological types can be found on their website under <a href="https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga7be549266bad7b2e6a04db49827f9f32" target="_blank" rel="noopener noreferrer">MorphTypes</a>. We will use `cv2.MORPH_CLOSE` for this example.
```
>>> kernel = np.ones((5, 5), np.uint8)
>>> closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
>>> showImage(closed_edges)
```

Now the edges should be mostly closed-in. Play around with different thresholds for Canny and kernel sizes for the morphology operations to handle any difficult images.

<img src="img/closed_edge.png" style="max-width: 100%;" title="Seed edge closing once processed with cv2.morphologyEx(). Image is zoomed in to show the closed edges." alt="Seed edge closing once processed with cv2.morphologyEx(). Image is zoomed in to show the closed edges.">



## Finding Object Contours
Now that we have all of our edges in a file, we can find the contours of the continous edges in the object. Basically, a contour is a continous edge with the same pixel intensity, hence why finding contours of a binary edge image is much easier than the raw image since all pixels are either 0 or 255 valued on a single binary channel. In OpenCV, we implement this by using the `cv2.findCountours()` function which takes 3 main arguments: the source image, a method for retrieving contours, and a curve approximation method. This function returns 2 objects: a tuple of of contour arrays, and the contour hierarchy which gives you the structure of nested contours if so desired. In this example, we only want the most external contour of each seed so we will set this to return only the highest leveled contour coordinates by setting the retrieval method to `cv2.RETR_EXTERNAL`.

```
>>> contours, _ = cv2.findContours(closed_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
>>> print(contours)
(array([[[3580, 3297]],

       [[3580, 3302]],

       [[3581, 3303]],

       [[3581, 3306]],
       ...

       [[2256,  701]],

       [[2256,  700]],

       [[2254,  700]],

       [[2253,  699]]], dtype=int32))

>>> print(len(contours))
110
```
So there are a total of 110 contours identified in the image. Let's clean up our code by defining a new function called `closedContours()` that will combine all three steps for us.
```
def closedContours(src: np.ndarray, threshold1: int, threshold2: int , kernel_size: int) -> tuple:
    """
    Take a source image
    Finds the image edges using the Canny algorithm
    Closes edge gaps with morphology close operations
    Finds all object external contours
    :Return: a tuple of contours
    """
    assert type(src)==np.ndarray
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.Canny(src, threshold1, threshold2)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

cnt = closedContours(img, 100, 200, 5)
```

We have all of the contours now so we can superimpose these onto our original image to make sure they match up as expected.
```
cnt_img = cv2.drawContours()
```

