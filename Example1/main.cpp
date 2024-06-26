#include "pch.h"
using namespace BNN;
// Tenarr(C,W,H,N) - (col major, channels are continuous)

int main(int argc, char* argv[]) {
	// Inference code
#if 1
	
#if 0
	// Either a IO program
	//Inference code
	NNet corr("corrective_cnn");
	NNet upsc("upscaling_cnn");
	if(!corr.Valid() || !upsc.Valid()) return 1;
	std::string img_name; double fact = 2;
	while(1) {
		print("Img path: "); std::cin >> img_name;
		if(img_name == "exit") return 0;
		print("Factor: "); std::cin >> fact;
		Tensor in = Image(img_name, 3, 1).tensor_rgb(true);
		if(in.size() == 0) { println("Invalid image!"); continue; }
		if(fact <= 0) { fact = 1; }
		Tensor out = corr.Compute_DS(in) + upsc.Compute_DS(in);
		double actual = 2;
		for(actual = 2; actual < fact; actual *= 2) {
			out = corr.Compute_DS(out) + upsc.Compute_DS(out);
		}
		int ext = img_name.find_last_of(".");
		img_name = img_name.substr(0, ext);
		Image(resize(in, fact, fact, BNN::Nearest)).save(img_name + "_nea.png");
		Image(resize(in, fact, fact, BNN::Linear)).save(img_name + "_lin.png");
		Image(resize(in, fact, fact, BNN::Cubic)).save(img_name + "_cub.png");
		Image(resize(in, fact, fact, BNN::Lanczos)).save(img_name + "_lc3.png");
		Image(resize(out, fact / actual, fact / actual, BNN::Cubic)).save(img_name + "_cnn.png");
	}
	
#else
	// Or a CLI
	NNet corr("corrective_cnn", 0);
	NNet upsc("upscaling_cnn", 0);
	if(argc == 1) {
		println("No arguments provided !");
		return -1;
	}
	if(!corr.Valid() || !upsc.Valid()) {
		println("Could not find network files !");
		return -1;
	}
	std::string img_name; 
	constexpr double fact = 2;
	for(int i = 1; i < argc; i++){
		img_name = argv[i];
		Tensor in = Image(img_name, 3, 0).tensor_rgb(true);
		if(in.size() == 0) { continue; }
		Tensor out = corr.Compute_DS(in) + upsc.Compute_DS(in);
		int ext = img_name.find_last_of(".");
		img_name = img_name.substr(0, ext);
		Image(resize(in, fact, fact, BNN::Nearest)).save(img_name + "_nea.png");
		Image(resize(in, fact, fact, BNN::Linear)).save(img_name + "_lin.png");
		Image(resize(in, fact, fact, BNN::Cubic)).save(img_name + "_cub.png");
		Image(resize(in, fact, fact, BNN::Lanczos)).save(img_name + "_lc3.png");
		Image(out).save(img_name + "_cnn.png");
	}
#endif
	// Training code
#else
	//Training code
	constexpr idx factor = 2;
	constexpr idx train_set = 500;
	constexpr idx test_set = 20;
	std::string parent = "Upscaler/";
	//Input data
	std::string in_folder = "Lowres/";
	std::string net_folder = argc > 1 ? argv[1] : "c5x32_c3x32x3_2x3";
	idx epochs = argc > 2 ? atoi(argv[2]) : 20;
	idx batch_sz = argc > 3 ? atoi(argv[3]) : 64;
	float  lrate = argc > 4 ? atof(argv[4]) : 0.0003f;
	std::string netname = parent + net_folder; //c5x32_c3x32x3_lin
	NNet upscl(parent + "c3_2x");
	Tenarr x(3, 240, 160, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		x.chip(i, 3) = Image(parent + in_folder + std::to_string(i), 3).tensor_rgb();
	//Output data
	std::string out_folder = "Reference/";
	Tenarr y(3, 480, 320, train_set);
#pragma omp parallel for
	for(idx i = 0; i < train_set; i++)
		y.chip(i, 3) = Image(parent + out_folder + std::to_string(i), 3).tensor_rgb()
		//- resize(x.chip(i, 3), 2, 2); 
		- upscl.Compute(x.chip(i, 3));
	//Test data
	std::string test_folder = "Test_lr/";
	Tenarr z(3, 240, 160, test_set);
#pragma omp parallel for
	for(idx i = 0; i < test_set; i++)
		z.chip(i, 3) = Image(parent + test_folder + std::to_string(i), 3).tensor_rgb();

#if 0
	//hidden layers
	vector<Layer*> top;
	top.push_back(new Input(shp3(3, 240, 160)));
	top.push_back(new Conv(32, 5, 1, 2, top.back(), false, Afun::t_lrelu));
	top.push_back(new Conv(32, 3, 1, 1, top.back(), false, Afun::t_lrelu));
	top.push_back(new Conv(32, 3, 1, 1, top.back(), false, Afun::t_lrelu));
	top.push_back(new Conv(32, 3, 1, 1, top.back(), false, Afun::t_lrelu));
	top.push_back(new Conv(3 * factor * factor, 3, 1, 1, top.back(), false, Afun::t_lin));
	//top.push_back(new PixShuf(top.back(), 4));
	//top.push_back(new Conv(3, 2, 2, 0, top.back(), false, Afun::t_lin));
	//top.push_back(new Output(top.back(), Efun::t_mae));
	top.push_back(new OutShuf(top.back(), factor, Efun::t_mae));
	auto opt = new Adam(0.002f);

	NNet net(top, opt, netname);
#else
	NNet net(netname, true);
#endif
	constexpr int iter = 20;
	for(idx i = 0; i < iter; i++) {
		if(!net.Train_Minibatch(x, y, epochs, batch_sz, 16, -1, decay_rate(lrate, i, iter))) break;
		net.Save();
#pragma omp parallel for
		for(idx j = 0; j < test_set; j++) {
			//Image(resize(z.chip(j, 3), factor, factor)).save("Upscaler/Test_cu/" + std::to_string(j) + ".png");
			//Image(net.Compute(z.chip(j, 3))).save(netname + "/" + std::to_string(j) + ".png");
			//Image(net.Compute(z.chip(j, 3)) + resize(z.chip(j, 3), factor, factor)).save(netname + "/" + std::to_string(j) + ".png");
			Image(net.Compute(z.chip(j, 3)) + upscl.Compute(z.chip(j, 3))).save(netname + "/" + std::to_string(j) + ".png");
		}
		//net.Save_images(z);
}
#endif

}
