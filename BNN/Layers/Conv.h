#pragma once
#include "Layer.h"
namespace BNN {
	//Normal convolution
	class Conv : public Layer {
	public:
		Conv() {}
		Conv(shp3 din, idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev = nullptr, bool bias = true, Afun af = Afun::t_lrelu) :
			Layer({ nch, c_dim(din[1],ks[0],st[0],pa[0]), c_dim(din[2],ks[1],st[1],pa[1]) }, prev),
			dz(odims()), b(bias ? dim1<3>{nch, 1, 1} : dim1<3>{ 0,0,0 }), w(din[0] * nch, ks[0], ks[1]),
			din(din), ks(ks), st(st), pa(pa), af(af), bias(bias) {
			_init();
		}
		Conv(idx nch, shp2 ks, shp2 st, shp2 pa, Layer* prev, bool bias = true, Afun af = Afun::t_lrelu) :
			Layer({ nch, c_dim(prev->odim(1),ks[0],st[0],pa[0]), c_dim(prev->odim(2),ks[1],st[1],pa[1]) }, prev),
			dz(odims()), b(bias ? dim1<3>{nch, 1, 1} : dim1<3>{ 0,0,0 }), w(pdim(0)* nch, ks[0], ks[1]),
			din(pdims()), ks(ks), st(st), pa(pa), af(af), bias(bias) {
			_init();
		}
		void init() override { _init(); }
		void derivative(bool ptrain) override {
			dz = y() * dz.unaryExpr(af.dx());
			if(ptrain) conv2d_igrad(x(), dz, w, st, ks - pa - 1, 1);
			/*if(st[0] > 1 || st[1] > 1) {
				auto dy = dz.inflate(dim1<3>{ 1, st[0], st[1] });
				if(ptrain) rev_convolve(x(), dy, w, 1, ks - pa - 1);
			}
			else {
				if(ptrain) rev_convolve(x(), dz, w, 1, ks - pa - 1);
			}*/
		}
		void gradient(Tensor& dw, Tensor& db, bool ptrain) override {
			dz = y() * dz.unaryExpr(af.dx());
			if(bias)db += dz.sum(dim1<2>{1, 2});
			/*if(st[0] > 1 || st[1] > 1) { //Avoid creating temporary on the most common filter with stride 1 also larger stride might actually compensate for the inflated vector
				//necessary temporary :( but this might be actually faster than doing the inflation on fly (3 integer divs, prevents autovectorization), so kind of a win ?
				Tensor dy = dz.inflate(dim1<3>{ 1, st[0], st[1] });
				acc_convolve(dw, x(), dy, 1, pa);
				if(ptrain) rev_convolve(x(), dy, w, 1, ks - pa - 1);
			}
			else {
				acc_convolve(dw, x(), dz, 1, pa);
				if(ptrain) rev_convolve(x(), dz, w, 1, ks - pa - 1);
			}*/
			conv2d_wgrad(dw, x(), dz, st, pa, 1);
			if(ptrain) conv2d_igrad(x(), dz, w, st, ks - pa - 1, 1);
		}

		void print()const override {
			println("Conv\t|", "\tDim:", odim(0), odim(1), odim(2), "\tKernel:", ks[0], ks[1],
				"\tStride:", st[0], st[1], "\tPad:", pa[0], pa[1], "\tBias", bias, "\tAf:", af.name());
		}
		void save(std::ostream& out)const override {
			out << "Hidden Conv" SPC din[0] SPC din[1] SPC din[2] SPC odim(0) SPC ks[0]
				SPC ks[1] SPC st[0] SPC st[1] SPC pa[0] SPC pa[1] SPC bias SPC af.type << "\n";
			out.write((const char*)w.data(), 4 * w.size());
			if(bias)out.write((const char*)b.data(), 4 * b.size());
			out << "\n";
			next->save(out);
		}
		static auto load(std::istream& in) {
			shp3 d; shp2 k, s, p; idx n; idx af; bool b;
			in >> d[0] >> d[1] >> d[2] >> n >> k[0]
				>> k[1] >> s[0] >> s[1] >> p[0] >> p[1] >> b >> af; in.ignore(1, '\n');
			auto tmp = new Conv(d, n, k, s, p, nullptr, b, Afun::Type(af));
			in.read((char*)tmp->get_w()->data(), tmp->w.size() * 4);
			if(b)in.read((char*)tmp->get_b()->data(), tmp->b.size() * 4);
			return tmp;
		}
		Conv* clone() const override { return new Conv(*this); }
		Tensor* get_w() override { return &w; }
		Tensor* get_b() override { return &b; }
		dim1<3> wdims() const override { return w.dimensions(); }
		dim1<3> bdims() const override { return b.dimensions(); }
		dim1<3> idims() const override { return din; }
		LType type() const override { return t_Conv; }
	private:
		void _init() {
			if(bias)b.setZero();
			xavier_init(w, din[0] * ks[0] * ks[1], ks[0] * ks[1] * w.dimension(0) / din[0]);
		}
		Tensor compute(const Tensor& x) const override {
			if(bias)return next->compute((conv2df(x, w, st, pa, 1) + b.broadcast(dim1<3>{1, odim(1), odim(2)})).unaryExpr(af.fx()));
			else return next->compute(conv2df(x, w, st, pa, 1).unaryExpr(af.fx()));
			/*if(bias)return next->compute((conv(x, w, st, pa) + b.broadcast(dim1<3>{1, odim(1), odim(2)})).unaryExpr(af.fx()));
			else return next->compute(conv(x, w, st, pa).unaryExpr(af.fx()));*/
		}
		Tensor compute_ds(const Tensor& x) const override {
			/*if(bias) return next->compute_ds((conv(x, w, st, pa)
				+ b.broadcast(dim1<3>{1, c_dim(x.dimension(1), ks[0], st[0], pa[0]),
				c_dim(x.dimension(2), ks[1], st[1], pa[1])})).unaryExpr(af.fx()));
			else return next->compute_ds(conv(x, w, st, pa).unaryExpr(af.fx()));*/
			if(bias)return next->compute_ds((conv2df(x, w, st, pa, 1) + b.broadcast(dim1<3>{1, c_dim(x.dimension(1), ks[0], st[0], pa[0]),
				c_dim(x.dimension(2), ks[1], st[1], pa[1])})).unaryExpr(af.fx()));
			else return next->compute_ds(conv2df(x, w, st, pa, 1).unaryExpr(af.fx()));
		}
		const Tensor& predict() override {
			conv2d(dz, x(), w, st, pa, 1);
			//convolve(dz, x(), w, st, pa);
			if(bias)dz = dz + b.broadcast(dim1<3>{1, odim(1), odim(2)});
			y() = dz.unaryExpr(af.fx());
			return next->predict();
		}
		const Tensor& predict(const Tensor& x) override {
			conv2d(dz, x, w, st, pa, 1);
			//convolve(dz, x, w, st, pa);
			if(bias)dz = dz + b.broadcast(dim1<3>{1, odim(1), odim(2)});
			return next->predict(y() = dz.unaryExpr(af.fx()));
		}
		Tensor dz, b, w;
		dim1<3> din;
		shp2 ks, st, pa;
		Afun af;
		bool bias = true;
	};
}