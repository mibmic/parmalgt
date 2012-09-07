#include "Background.h"
#include "gtest/gtest.h"
#include "Point.hpp"
#include "Geometry.hpp"

const int DIM = 4;
const int SIZE = 8;


TEST(Geometry, Neighbors){

  geometry::Geometry<DIM>::extents_t e;
  std::fill(e.begin(), e.end(), SIZE);
  geometry::Geometry<DIM> g(e);
  geometry::Geometry<DIM>::raw_pt_t n;
  std::fill(n.begin(), n.end(), 0);
  for (; n[0] < SIZE; ++n[0])
    for (; n[1] < SIZE; ++n[1])
      for (; n[2] < SIZE; ++n[2])
        for (; n[3] < SIZE; ++n[3])
          for (pt::Direction<DIM> mu; mu.is_good(); ++mu){
            geometry::Geometry<DIM>::raw_pt_t x(n);
            x[mu]++;
            ASSERT_EQ(g.mk_point(x), g.mk_point(n) + mu);
            ASSERT_NE(g.mk_point(x), g.mk_point(n));
            x[mu] -= 2;
            ASSERT_EQ(g.mk_point(x), g.mk_point(n) - mu);
            ASSERT_NE(g.mk_point(x), g.mk_point(n));
        }
};

TEST(Geometry, ManualVsDirectionAccess){

  geometry::Geometry<DIM>::extents_t e;
  std::fill(e.begin(), e.end(), SIZE);
  geometry::Geometry<DIM> g(e);
  geometry::Geometry<DIM>::raw_pt_t n;
  std::fill(n.begin(), n.end(), 0);
  pt::Point<DIM> x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x += pt::Direction<DIM>(0))
    for (; n[1] < SIZE; ++n[1], x += pt::Direction<DIM>(1))
      for (; n[2] < SIZE; ++n[2], x += pt::Direction<DIM>(2))
        for (; n[3] < SIZE; ++n[3], x += pt::Direction<DIM>(3))
            ASSERT_EQ(x, g.mk_point(n));

};




TEST(Geometry, Periodicity){
  geometry::Geometry<DIM>::extents_t e;
  std::fill(e.begin(), e.end(), SIZE);
  geometry::Geometry<DIM> g(e);
  geometry::Geometry<DIM>::raw_pt_t n;
  std::fill(n.begin(), n.end(), 0);
  pt::Point<DIM> x_0 = g.mk_point(n);
  pt::Point<DIM> x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x += pt::Direction<DIM>(0))
    for (; n[1] < SIZE; ++n[1], x += pt::Direction<DIM>(1))
     for (; n[2] < SIZE; ++n[2], x += pt::Direction<DIM>(2))
       for (; n[3] < SIZE; ++n[3], x += pt::Direction<DIM>(3)) ;
  ASSERT_EQ(x, x_0);
  std::fill(n.begin(), n.end(), 0);
  x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x += pt::Direction<DIM>(0));
  ASSERT_EQ(x, x_0);
  x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x += pt::Direction<DIM>(1));
  ASSERT_EQ(x, x_0);
  x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x += pt::Direction<DIM>(2));
  ASSERT_EQ(x, x_0);
    x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x += pt::Direction<DIM>(3));
  ASSERT_EQ(x, x_0);
  x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x -= pt::Direction<DIM>(0));
  ASSERT_EQ(x, x_0);
  x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x -= pt::Direction<DIM>(1));
  ASSERT_EQ(x, x_0);
  x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x -= pt::Direction<DIM>(2));
  ASSERT_EQ(x, x_0);
    x = g.mk_point(n);
  for (; n[0] < SIZE; ++n[0], x -= pt::Direction<DIM>(3));
  ASSERT_EQ(x, x_0);
}
