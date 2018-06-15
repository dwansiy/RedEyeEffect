import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.CvType.CV_8U;

public class RedEyeEffect {
    private static final CascadeClassifier eyeClassifier;

    private static final String redEyeImage = "red1.png";

    private static final int WINDOW_WIDTH = 600;
    private static final int WINDOW_HEIGHT = 300;

    private static final int SOURCE_WIDTH = 650;
    private static final int SOURCE_HEIGHT = 550;
    private static final int RESULT_WIDTH = 300;
    private static final int RESULT_HEIGHT = 300;

    private static JFrame sourceJFrame;
    private static JLabel sourceJLabel;

    private static JFrame resultJFrame;
    private static JLabel resultJLabel;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        eyeClassifier = new CascadeClassifier("data\\haarcascade files\\haarcascade_eye_tree_eyeglasses.xml");
    }

    public static void main(String[] args) {
        Mat sourceMat = Imgcodecs.imread(redEyeImage);
        showMat(sourceMat, "원본");

        Mat blurMat = sourceMat.clone();
        Imgproc.GaussianBlur(blurMat, blurMat, new Size(9, 9), 1.26);
        showMat(blurMat, "블러");

        Mat eyeMat = sourceMat.clone();
        Rect[] eyes = findEyes(eyeMat);
        drawRectangle(eyeMat, eyes);
        showMat(eyeMat, "눈 탐지");

        Mat[] eyesMask = createMask(blurMat, eyes);

        for (int i = 0; i < eyesMask.length; i++) {
            showMat(eyesMask[i], i + "번째 눈 마스크");
            eyesMask[i] = floodFill(eyesMask[i]);
            Imgproc.dilate(eyesMask[i], eyesMask[i], new Mat(new Size(3, 3), CV_8U, new Scalar(1)), new Point(-1, -1), 3);
            showMat(eyesMask[i], i + "번째 눈 마스크 flood fill");
        }


        Mat[] eyesMat = new Mat[eyes.length];
        for (int i = 0; i < eyesMat.length; i++) {
            eyesMat[i] = swapColor(sourceMat.submat(eyes[i]), eyesMask[i]);
        }
        Mat resultMat = sourceMat.clone();
        for (Mat eye : eyesMat) {
            resultMat.copyTo(eye);
        }
        showMat(resultMat, "결과물");
    }

    private static Rect[] findEyes(Mat source) {
        MatOfRect eyes = new MatOfRect();
        eyeClassifier.detectMultiScale(source, eyes);
        return eyes.toArray();
    }

    private static void drawRectangle(Mat source, Rect[] eyes) {
        if (eyes == null || eyes.length == 0) return;

        for (Rect rect : eyes) {
            Imgproc.rectangle(source, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(50, 200, 50), 1);
        }
    }

    private static Mat[] createMask(Mat mat, Rect[] eyes) {
        Mat[] maskArr = new Mat[eyes.length];
        for (int i = 0; i < eyes.length; i++) {
            maskArr[i] = mat.submat(eyes[i]);
            List<Mat> channelMats = new ArrayList<>();
            Core.split(maskArr[i], channelMats);
            Mat redMat = channelMats.get(2);
            Mat bgMat = new Mat();
            Core.add(channelMats.get(0), channelMats.get(1), bgMat);
            Mat mask = Mat.zeros(redMat.rows(), redMat.cols(), CV_8U);
            final int redThreshold = 70;
            for (int j = 0; j < redMat.cols(); j++) {
                for (int k = 0; k < redMat.rows(); k++) {
                    double[] doubles = redMat.get(k, j);
                    double[] tmp = new double[doubles.length];
                    if ((doubles[0] > redThreshold) && (doubles[0] > bgMat.get(k, j)[0])) {
                        tmp[0] = 255;
                    } else {
                        tmp[0] = 0;
                    }
                    mask.put(k, j, tmp);
                }
            }
            maskArr[i] = mask;
        }
        return maskArr;
    }

    private static Mat floodFill(Mat mat) {
        Mat mask = mat.clone();
        Mat tmp = Mat.zeros(mask.rows() + 2, mask.cols() + 2, mask.type());
        Imgproc.floodFill(mask, tmp, new Point(0, 0), new Scalar(255, 255, 255));
        Core.bitwise_not(mask, mask);
        Mat result = new Mat();
        Core.bitwise_or(mask, mat, result);
        return result;
    }

    private static Mat swapColor(Mat mat, Mat mask) {
        List<Mat> channels = new ArrayList<>();
        Core.split(mat, channels);
        for (int j = 0; j < mat.cols(); j++) {
            for (int k = 0; k < mat.rows(); k++) {
                if (mask.get(k, j)[0] == 255) {
                    double mean = (channels.get(0).get(k, j)[0] + channels.get(1).get(k, j)[0]) / 2;
                    double[] temp = new double[1];
                    temp[0] = mean;
                    channels.get(0).put(k, j, temp);
                    channels.get(1).put(k, j, temp);
                    channels.get(2).put(k, j, temp);
                }
            }
        }
        Core.merge(channels, mat);
        return mat;
    }

    private static void processingVideo(Mat source) throws IOException {
        Rect[] eyes = findEyes(source);
        Mat clone = source.clone();
        drawRectangle(clone, eyes);

        BufferedImage image = convertMat(clone);
        showOriginal(image);

        if (eyes != null && eyes.length != 0) {
            Mat resultMat = resize(new Mat(source, eyes[0]));

            Mat mat = new Mat();
            Imgproc.medianBlur(resultMat, mat, 21);
            //Imgproc.equalizeHist(mat,mat);
            //Imgproc.createCLAHE(5, new Size(5,5)).apply(mat,mat);

            //Imgproc.threshold(mat,mat,50,255,Imgproc.THRESH_BINARY_INV);
            //Imgproc.adaptiveTxhreshold(resultMat, mat, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 101, 1.1);
            //Imgproc.Canny(resultMat, mat, 10, 40);
            //Mat mask = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5), new Point(1, 1));
            //Mat mask = new Mat(new Size(3, 3), CV_8U, new Scalar(1));
            //Mat dilated_edges = new Mat();
            //Imgproc.morphologyEx(mat, dilated_edges, Imgproc.MORPH_CLOSE, mask, new Point(-1, -1), 1);
            //Imgproc.dilate(mat, mat, mask, new Point(-1, -1), 4);
            //List<MatOfPoint> contours = new ArrayList<>();
            //Imgproc.findContours(mat, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
            //for (int i = 0; i < contours.size(); i++) {
            //    double epsilon = Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true) * 0.03;
            //    MatOfPoint2f approx = new MatOfPoint2f();
            //    Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i).toArray()), approx, epsilon, true);
            //Imgproc.fillConvexPoly(mat,new MatOfPoint(approx.toArray()),new Scalar(255,255,255));
            //List<MatOfPoint> points = new ArrayList<>();
            //points.add(new MatOfPoint(approx.toArray()));
            //Imgproc.polylines(mat,points,true,new Scalar(255,255,255),2);
            //    Imgproc.drawContours(mat,contours, i, new Scalar(255, 255, 255),5);
            //}

            MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
            FastFeatureDetector fastFeatureDetector = FastFeatureDetector.create();
            fastFeatureDetector.setType(FastFeatureDetector.TYPE_9_16);
            fastFeatureDetector.detect(mat, matOfKeyPoint);
            //Features2d.drawKeypoints(mat, matOfKeyPoint, mat);

            KeyPoint[] keyPoints = matOfKeyPoint.toArray();
            if (keyPoints != null && keyPoints.length != 0) {
                int x = 0;
                int y = 0;
                int size = keyPoints.length;
                for (KeyPoint keyPoint : keyPoints) {
                    x += keyPoint.pt.x;
                    y += keyPoint.pt.y;
                }
                x = x / size;
                y = y / size;
                Point point = new Point(x, y);
                //Imgproc.drawMarker(mat, point, new Scalar(255, 255, 255));
            }

            //std::vector < std::vector < cv::Point > >contours;
            //cv::findContours (dilated_edges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
            //cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true) * 0.02, true);

            BufferedImage resultBufferedImage = convertMat(mat);
            showResult(resultBufferedImage);
        }
    }

    public static MatOfPoint convertIndexesToPoints(MatOfPoint contour, MatOfInt indexes) {
        int[] arrIndex = indexes.toArray();
        Point[] arrContour = contour.toArray();
        Point[] arrPoints = new Point[arrIndex.length];

        for (int i = 0; i < arrIndex.length; i++) {
            arrPoints[i] = arrContour[arrIndex[i]];
        }

        MatOfPoint hull = new MatOfPoint();
        hull.fromArray(arrPoints);
        return hull;
    }

    private static BufferedImage convertMat(Mat mat) throws IOException {
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, matOfByte);
        byte[] byteArray = matOfByte.toArray();
        InputStream in = new ByteArrayInputStream(byteArray);
        return ImageIO.read(in);
    }


    /*
    private static BufferedImage convertMat(Mat original) {
        BufferedImage image;
        int width = original.width(), height = original.height(), channels = original.channels();
        byte[] sourcePixels = new byte[width * height * channels];
        original.get(0, 0, sourcePixels);

        if (original.channels() > 1)
            image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
         else
            image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.length);

        return image;
    }
    */

    private static void showMat(Mat mat, String title) {
        showMat(mat, WINDOW_WIDTH, WINDOW_HEIGHT, title);
    }

    private static void showMat(Mat mat, int width, int height, String title) {
        BufferedImage image = null;
        try {
            image = convertMat(mat);
        } catch (IOException e) {
            //잘못된 mat 형식
        }
        if (image == null) return;

        makeWindow(image, width, height, title);
    }

    private static void makeWindow(BufferedImage image, int width, int height, String title) {
        JFrame frame = new JFrame();
        frame.setTitle(title);
        frame.setLayout(new FlowLayout());
        frame.setSize(width, height);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        JLabel label = new JLabel();
        ImageIcon imageIcon = new ImageIcon(image);
        label = new JLabel(imageIcon);
        frame.add(label);
        frame.revalidate();
    }

    private static void initUI() {
        sourceJFrame = new JFrame();
        sourceJFrame.setLayout(new FlowLayout());
        sourceJFrame.setSize(SOURCE_WIDTH, SOURCE_HEIGHT);
        sourceJFrame.setVisible(true);
        sourceJFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        resultJFrame = new JFrame();
        resultJFrame.setLayout(new FlowLayout());
        resultJFrame.setSize(RESULT_WIDTH, RESULT_HEIGHT);
        resultJFrame.setVisible(true);
        resultJFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

    private static Mat resize(Mat mat) {
        Mat result = new Mat();
        Imgproc.resize(mat, result, new Size(RESULT_WIDTH, RESULT_HEIGHT));
        return result;
    }

    private static void showOriginal(Image img) {
        if (sourceJLabel != null)
            sourceJFrame.remove(sourceJLabel);
        ImageIcon imageIcon = new ImageIcon(img);
        sourceJLabel = new JLabel(imageIcon);
        sourceJFrame.add(sourceJLabel);
        sourceJFrame.revalidate();
    }

    private static void showResult(Image img) {
        if (resultJLabel != null)
            resultJFrame.remove(resultJLabel);
        ImageIcon imageIcon = new ImageIcon(img);
        resultJLabel = new JLabel(imageIcon);
        resultJFrame.add(resultJLabel);
        resultJFrame.revalidate();
    }
}