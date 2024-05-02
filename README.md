# ZPO Projekt - Upscalování obrazu (backend)
Autor: Ondřej Áč (xacond00)  
Založeno na vlastní NN knihovně: https://github.com/Panjaksli/BNN, implementován Lanczos algoritmus, inferenční kód a pár nových funkcí.  
## Použití k upscalování
1) Stáhnout release  
2) Spuštení programu s CLI: `Upscaler.exe img1 img2 ... imgN`  
3) Ukládá stejnojmenné 2x zvětšené obrázky pomocí různých upscalovacích algoritmů s novými příponami např. "_lin.png" v tomto pořadí: Nearest, Bilinear, Bicubic, Lanczos, CNN  

Původní návod k použití knihovny pro trenování:  
## Interface
The interface is as simple as possible - create vector<Layer> and push input, hidden layers and output, create optimizer and then pass both to the network, it manages the given memory itself **(dont delete anything manually!)**.
```cpp
using namespace BNN;
vector<Layer*> top;
top.push_back(new Input(shp3(3, 240, 160)));
top.push_back(new Conv(12, 5, 1, 2, top.back(), true, Afun::t_cubl));
top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_cubl));
top.push_back(new Conv(12, 3, 1, 1, top.back(), true, Afun::t_cubl));
top.push_back(new OutShuf(top.back(), 2));
//Optimizer with: learning rate, regularizer 
auto opt = new Adam(0.001f, Regular(RegulTag::L2, 0.1f));
NNet net(top, opt, "Network name");
```
If you do any changes to the architecture after that, you need to compile it before running !\
For training you can then use the Train_single(), Train_parallel() or **Train_Minibatch()** functions, followed by Save() to save the network to a hybrid text/binary file.
Save_images() can be used to save the output tensor as png image(s), if it has 1 or 3 channels.
```cpp
//Channels, Width, Height, Count
Tenarr x(3, 240, 160, train_set);
Tenarr y(3, 240, 160, train_set);
for(int i = 0; i < 100; i++) {
  // input, target, epochs, minibatch size, threads, steps (minibatches per epoch), learning rate
	if(!net.Train_Minibatch(x, y, 20, 16, 16, -1, decay_rate(0.001f, i, 20))) break;
	net.Save();
	net.Save_images(z);
}
```
### How does it work ?
The model is trained on the error of reference image and low res image upscaled with bicubic interpolation:\
d(x) = f(x) - g(x),\
where: d(x) is error function, f(x) is full resolution image and g(x) is an approximation of f(x).\
This error is then added to the upscaled image during inference:\
f(x) = g(x) + d(x) = g(x) - g(x) + f(x) = f(x).\
This results in reconstruction of original image f(x).

