#include <iostream>
#include <torch/script.h> // One-stop header.
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <memory>

using namespace std;
using namespace cv;


torch::jit::script::Module load_model(string model_path)
{
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        cout << "here??" << endl;
        module.to(at::kCUDA);
        module.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        cout << e.what() << endl;
        
    }
    cout << "loading model OK" << "\n";
    return module;
}

torch::Tensor inference(torch::jit::script::Module module, torch::Tensor img_tensor)
{
    std::vector<torch::jit::IValue> inputs;
    torch::Tensor out;
    inputs.push_back(img_tensor);
    out = module.forward(std::move(inputs)).toTensor();
    return out;
}

torch::Tensor img2tensor(cv::Mat image)
{
    cv::Mat img_float;
    //cv::resize(image, image, cv::Size(128, 96));
    cout << img_float.size() << endl;
    image.convertTo(img_float, CV_32F, 1.0 / 255, 0);
    auto img_tensor = torch::from_blob(img_float.data, { 1, img_float.rows, img_float.cols,3 });
    cout << "from_blob : " << img_tensor.sizes() << endl;
    img_tensor = img_tensor.permute({ 0,3,1,2 });
    img_tensor = img_tensor.to(at::kCUDA);
    cout << "img_tensor : " << img_tensor.sizes() << endl;
    
    return img_tensor;
}

cv::Mat torchTensortoCVMat(torch::Tensor& tensor)
{
    cout << tensor.sizes() << endl;
    tensor = tensor.squeeze().detach();
    cout << tensor.sizes() << endl;
    tensor = tensor.permute({ 1, 2, 0 }).contiguous();
    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);
    int64_t height = tensor.size(0);
    int64_t width = tensor.size(1);
    cout << width << " " << height << endl;
    cv::Mat mat = cv::Mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr<uchar>());
    return mat.clone();
}


int sr_image()
{
    //����ģ��
    string weight_path = "D:/workroom/project/riverlight/HSR/weights/BSRGANx2.pt";
    //string weight_path = "D:/workroom/project/riverlight/qir/weights/HSRx2_m1_5_10.pt";
    torch::jit::script::Module module = load_model(weight_path);

    //����ͼ��
    string img_path = "d:/workroom/testroom/old.png";
    cv::Mat image = cv::imread(img_path);
    auto img_tensor = img2tensor(image);
    cout << "img2tensor done " << endl;
    cout << "shape : " << img_tensor.sizes() << endl;

    // evalute time
    double t, t0;
    int num = 1;
    torch::Tensor out;
    t0 = (double)cv::getTickCount();

    cout << "start loop " << endl;
    for (int i = 0; i < num; i++)
    {
        t = (double)cv::getTickCount();
        auto pred = inference(module, img_tensor);
        cout << "out shape : " << pred.sizes() << endl;
        t = (double)cv::getTickCount() - t;
        printf("���źķ�ʱ��Ϊ:  %gs\n", t / cv::getTickFrequency());
        cv::Mat out_img = torchTensortoCVMat(pred);
        cv::imwrite("d:/out.png", out_img);
    }
    cout << "loop done " << endl;

    //std::cout << out << std::endl;
    //std::cout << pred << std::endl;

    t0 = (double)cv::getTickCount() - t0;
    printf("�ܺķ�ʱ��Ϊ:  %gs\n", t0 / cv::getTickFrequency());
    printf("ÿ��:  %g �� \n", num * cv::getTickFrequency() / t0);

    std::cout << "ok\n";

    return 0;
}

int sr_video()
{
    VideoCapture cap("d:/workroom/testroom/xgm_lr.mp4");
    VideoWriter write;
    write.open("d:/workroom/testroom/xgm_bsrgan.avi", /*CAP_OPENCV_MJPEG*/ VideoWriter::fourcc('I', '4', '2', '0'),
        cap.get(cv::CAP_PROP_FPS), Size(cap.get(cv::CAP_PROP_FRAME_WIDTH)*2, cap.get(cv::CAP_PROP_FRAME_HEIGHT)*2));

    //����ģ��
    string weight_path = "D:/workroom/project/riverlight/HSR/weights/BSRGANx2.pt";
    torch::jit::script::Module module = load_model(weight_path);

    while (1) {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        auto img_tensor = img2tensor(frame);
        auto pred = inference(module, img_tensor);
        cv::Mat out_img = torchTensortoCVMat(pred);

        write << out_img;
        cout << cap.get(CAP_PROP_POS_MSEC) << endl;
    }
    cap.release();
    write.release();

    cout << "done" << endl;

    return 0;
}


int main(int argc, const char* argv[]) 
{
    sr_image();
    //sr_video();

    return 0;
}
