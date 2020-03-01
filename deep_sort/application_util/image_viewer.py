"""
This module contains an image viewer and drawing routines based on OpenCV.
"""
import numpy as np
import cv2
import time


def is_in_bounds(mat, roi):
    """Check if ROI is fully contained in the image.

    Parameters
    ----------
    mat : ndarray
        An ndarray of ndim>=2.
    roi : (int, int, int, int)
        Region of interest (x, y, width, height) where (x, y) is the top-left
        corner.

    Returns
    -------
    bool
        Returns true if the ROI is contain in mat.

    """
    if roi[0] < 0 or roi[0] + roi[2] >= mat.shape[1]:
        return False
    if roi[1] < 0 or roi[1] + roi[3] >= mat.shape[0]:
        return False
    return True


def view_roi(mat, roi):
    """Get sub-array.

    The ROI must be valid, i.e., fully contained in the image.

    Parameters
    ----------
    mat : ndarray
        An ndarray of ndim=2 or ndim=3.
    roi : (int, int, int, int)
        Region of interest (x, y, width, height) where (x, y) is the top-left
        corner.

    Returns
    -------
    ndarray
        A view of the roi.

    """
    sx, ex = roi[0], roi[0] + roi[2]
    sy, ey = roi[1], roi[1] + roi[3]
    if mat.ndim == 2:
        return mat[sy:ey, sx:ex]
    else:
        return mat[sy:ey, sx:ex, :]


class ImageViewer(object):
    """An image viewer with drawing routines and video capture capabilities.

    Key Bindings:

    * 'SPACE' : pause
    * 'ESC' : quit

    Parameters
    ----------
    update_ms : int
        Number of milliseconds between frames (1000 / frames per second).
    window_shape : (int, int)
        Shape of the window (width, height).
    caption : Optional[str]
        Title of the window.

    Attributes
    ----------
    image : ndarray
        Color image of shape (height, width, 3). You may directly manipulate
        this image to change the view. Otherwise, you may call any of the
        drawing routines of this class. Internally, the image is treated as
        beeing in BGR color space.

        Note that the image is resized to the the image viewers window_shape
        just prior to visualization. Therefore, you may pass differently sized
        images and call drawing routines with the appropriate, original point
        coordinates.
    color : (int, int, int)
        Current BGR color code that applies to all drawing routines.
        Values are in range [0-255].
    text_color : (int, int, int)
        Current BGR text color code that applies to all text rendering
        routines. Values are in range [0-255].
    thickness : int
        Stroke width in pixels that applies to all drawing routines.

    """

    def __init__(self, update_ms, window_shape=(640, 480), caption="Figure 1"):
        self._window_shape = window_shape
        self._caption = caption
        self._update_ms = update_ms
        self._video_writer = None
        self._user_fun = lambda: None
        self._terminate = False

        self.image = np.zeros(self._window_shape + (3, ), dtype=np.uint8)
        self._color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.thickness = 1

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("color must be tuple of 3")
        self._color = tuple(int(c) for c in value)

    def rectangle(self, x, y, w, h, label=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.

        """
        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        if label is not None:
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 1, self.thickness)

            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                text_size[0][1]
            cv2.rectangle(self.image, pt1, pt2, self._color, -1)
            cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255), self.thickness)

    def circle(self, x, y, radius, label=None):
        """Draw a circle.

        Parameters
        ----------
        x : float | int
            Center of the circle (x-axis).
        y : float | int
            Center of the circle (y-axis).
        radius : float | int
            Radius of the circle in pixels.
        label : Optional[str]
            A text label that is placed at the center of the circle.

        """
        image_size = int(radius + self.thickness + 1.5)  # actually half size
        roi = int(x - image_size), int(y - image_size), \
            int(2 * image_size), int(2 * image_size)
        if not is_in_bounds(self.image, roi):
            return

        image = view_roi(self.image, roi)
        center = image.shape[1] // 2, image.shape[0] // 2
        cv2.circle(
            image, center, int(radius + .5), self._color, self.thickness)
        if label is not None:
            cv2.putText(
                self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
                2, self.text_color, 2)

    def gaussian(self, mean, covariance, label=None):
        """Draw 95% confidence ellipse of a 2-D Gaussian distribution.

        Parameters
        ----------
        mean : array_like
            The mean vector of the Gaussian distribution (ndim=1).
        covariance : array_like
            The 2x2 covariance matrix of the Gaussian distribution.
        label : Optional[str]
            A text label that is placed at the center of the ellipse.

        """
        # chi2inv(0.95, 2) = 5.9915
        vals, vecs = np.linalg.eigh(5.9915 * covariance)
        indices = vals.argsort()[::-1]
        vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

        center = int(mean[0] + .5), int(mean[1] + .5)
        axes = int(vals[0] + .5), int(vals[1] + .5)
        angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
        cv2.ellipse(
            self.image, center, axes, angle, 0, 360, self._color, 2)
        if label is not None:
            cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
                        2, self.text_color, 2)

    def run(self, update_fun=None):
        """Start the image viewer.

        This method blocks until the user requests to close the window.

        Parameters
        ----------
        update_fun : Optional[Callable[] -> None]
            An optional callable that is invoked at each frame. May be used
            to play an animation/a video sequence.

        """
        if update_fun is not None:
            self._user_fun = update_fun

        self._terminate, is_paused = False, False
        # print("ImageViewer is paused, press space to start.")
        while not self._terminate:
            t0 = time.time()
            if not is_paused:
                self._terminate = not self._user_fun()
                if self._video_writer is not None:
                    self._video_writer.write(
                        cv2.resize(self.image, self._window_shape))
            t1 = time.time()
            remaining_time = max(1, int(self._update_ms - 1e3*(t1-t0)))
            cv2.imshow(
                self._caption, cv2.resize(self.image, self._window_shape[:2]))
            key = cv2.waitKey(remaining_time)


        # Due to a bug in OpenCV we must call imshow after destroying the
        # window. This will make the window appear again as soon as waitKey
        # is called.
        #
        # see https://github.com/Itseez/opencv/issues/4535
        self.image[:] = 0
        cv2.destroyWindow(self._caption)
        cv2.waitKey(1)
        cv2.imshow(self._caption, self.image)

    def stop(self):
        """Stop the control loop.

        After calling this method, the viewer will stop execution before the
        next frame and hand over control flow to the user.

        Parameters
        ----------

        """
        self._terminate = True
