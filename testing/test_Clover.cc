#include <Clover.hpp>
#include <gtest/gtest.h>
#include <LocalField.hpp>
#include <Background.h>
#include <Geometry.hpp>
#include <algorithm>
#include <newQCDpt.h>
#include <Helper.h>

// space-time dimensions
const int DIM = 4;
// perturbative order
const int ORD = 2;

typedef bgf::AbelianBgf Bgf_t; // background field
typedef BGptSU3<Bgf_t, ORD> ptSU3; // group variables
typedef ptt::PtMatrix<ORD> ptsu3; // algebra variables
typedef BGptGluon<Bgf_t, ORD, DIM> ptGluon; // gluon
typedef pt::Point<DIM> Point;
typedef pt::Direction<DIM> Direction;

// shorthand for gluon field
typedef fields::LocalField<ptGluon, DIM> GluonField;
typedef GluonField::neighbors_t nt;

template <class Field_t>
struct randomize {
  randomize(const int& seed) { std::srand(seed); }
  static const int dim = Field_t::dim;
  void operator()(Field_t& F, const pt::Point<dim>& n) const {
    F[n].randomize();
  }
  static const int n_cb = 0;
};


template <class F>
struct compare {
  static const int n_cb = 0;

  void operator()(F& fld, const pt::Point<DIM>& n) const {
    clover::detail::FieldWrapper<F> fwrp(fld);
    for (pt::Direction<DIM> mu; mu.is_good(); ++mu){
      ASSERT_TRUE(fld[n][mu].bgf() == fwrp(n, mu).bgf());
      ASSERT_TRUE(dag(fld[n-mu][mu]).bgf() == fwrp(n, -mu).bgf());
      for (int i = 0; i < ORD; ++i){
	ASSERT_TRUE(SU3Cmp(fwrp(n, mu)[i], fld[n][mu][i])());
	ASSERT_TRUE(SU3Cmp(fwrp(n, -mu)[i], (fld[n - mu][mu][i]).dag())());
      }
    }
  }
};

template <class F>
struct pl_compare {
  static const int n_cb = 0;

  void operator()(F& fld, const pt::Point<DIM>& n) const {
    clover::detail::FieldWrapper<F> fwrp(fld);
    for (pt::Direction<DIM> mu; mu.is_good(); ++mu)
      for (pt::Direction<DIM> nu; nu.is_good(); ++nu){
	typename kernels::std_types<F>::ptSU3_t 
	  tmpff = // forward-forward
	  fld[n][mu] * fld[n+mu][nu] * 
	  dag(fld[n+nu][mu]) * dag(fld[n][nu]),
	  tmpfb = // forward-backward
	  fld[n][mu] * dag(fld[n + mu - nu][nu]) *
	  dag(fld[n - nu][mu]) * fld[n - nu][nu],
	  tmpbf = // backward-forward
	  dag(fld[n - mu][mu]) * fld[n - mu][nu] *
	  fld[n - mu + nu][mu] * dag(fld[n][nu]),
	  tmpbb = // backward-backward
	  dag(fld[n - mu][mu]) * dag(fld[n - mu - nu][nu]) *
	  fld[n - mu - nu][mu] * fld[n - nu][nu],
	  ff = fwrp(n, mu, nu),
	  fb = fwrp(n, mu, -nu),
	  bf = fwrp(n, -mu, nu),
	  bb = fwrp(n, -mu, -nu);
      ASSERT_TRUE(tmpff.bgf() == ff.bgf());
      ASSERT_TRUE(tmpfb.bgf() == fb.bgf());
      ASSERT_TRUE(tmpbf.bgf() == bf.bgf());
      ASSERT_TRUE(tmpbb.bgf() == bb.bgf());
      for (int i = 0; i < ORD; ++i){
	ASSERT_TRUE(SU3Cmp(ff[i], tmpff[i])());
	ASSERT_TRUE(SU3Cmp(fb[i], tmpfb[i])());
	ASSERT_TRUE(SU3Cmp(bb[i], tmpbb[i])());
	ASSERT_TRUE(SU3Cmp(bf[i], tmpbf[i])());
      }
    }
  }
};

TEST(util, wrapper){
  geometry::Geometry<DIM>::extents_t e;
  // we want a L = 4 lattice
  int T = 10;
  int L = T;
  int s = 0;
  std::fill(e.begin(), e.end(), 5);
  // for SF boundary: set the time extend to T + 1
  e[0] = T + 1;
  // initialize background field get method
  bgf::get_abelian_bgf(0, 0, T, L, s);
  // we will have just one field
  GluonField U(e, 1, 0, nt());
  U.apply_everywhere(randomize<GluonField>(8134));
  U.apply_everywhere(compare<GluonField>());
  U.apply_everywhere(pl_compare<GluonField>());
}
