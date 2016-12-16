#pragma once

// This file contains opens the part of the namespace of the Eigen linear
// algebra library used by this project, so we don't have to do this every time

#include <cmath> // needed for the EigenTransformPlugin.h below
#include <utility>

// Plugin implementing custom tranformations
//#define EIGEN_MATRIXBASE_PLUGIN "EigenMatrixBasePlugin.h"

// #define EIGEN_USE_MKL_ALL

#undef B2
//#define EIGEN_HAS_OPENMP
#include <Eigen/Sparse>
#include <Eigen/LU>
#include <Eigen/Dense>

// Dense types
// to reduce verbosity explicitly state which Eigen structures we use here
using Tripletd = Eigen::Triplet<double>;
template<typename T>
using Triplet = Eigen::Triplet<T>;

using Eigen::MatrixBase;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::Matrix2d;
using Eigen::Matrix4d;
using Matrix4x2d = Eigen::Matrix<double, 4, 2>;
using Matrix4x5d = Eigen::Matrix<double, 4, 5>;

using VecXd = Eigen::VectorXd;
using VecXf = Eigen::VectorXf;
using Vec2d = Eigen::Vector2d;
using Vec2f = Eigen::Vector2f;
using Vec2i = Eigen::Vector2i;
using Vec2ui = Eigen::Matrix<unsigned int, 2, 1>;
using Vec3d = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Vec3i = Eigen::Vector3i;
using Vec3ui = Eigen::Matrix<unsigned int, 3, 1>;

using Vec4d = Eigen::Vector4d;
using Vec4f = Eigen::Vector4f;
using Vec4i = Eigen::Vector4i;
using Vec4ui = Eigen::Matrix<unsigned int, 4, 1>;

using Vec6i = Eigen::Matrix<int, 6, 1>;

template<typename T>
using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template<typename T>
using Vec2 = Eigen::Matrix<T, 2, 1>;
template<typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

// Sparse types
using SparseMatrixd = Eigen::SparseMatrix<double>;

template<typename T>
using SelfAdjointEigenSolver = Eigen::SelfAdjointEigenSolver<T>;
template<typename T>
using SparseLU = Eigen::SparseLU<Eigen::SparseMatrix<T>>;
template<typename T>
using SimplicialLDLT = Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>>;
template<typename T>
//using BiCGSTAB = Eigen::BiCGSTAB<Eigen::SparseMatrix<T>, Eigen::IncompleteLUT<T>>;
using BiCGSTAB = Eigen::BiCGSTAB<Eigen::SparseMatrix<T>>;

template<typename T>
using PCG = Eigen::ConjugateGradient<Eigen::SparseMatrix<T>, Eigen::Lower|Eigen::Upper>;

// convenience function to check for successful Eigen computation
template <typename Solver>
inline bool isSuccessful( Solver &s ) { return s.info() == Eigen::Success; }


// additional convenience functions
template<typename DerivedA, typename DerivedB>
inline typename DerivedA::Scalar
dist(const MatrixBase<DerivedA>& v1, const MatrixBase<DerivedB>& v2)
{ return (v1 - v2).norm(); }

template<typename DerivedA, typename DerivedB>
inline typename DerivedA::Scalar
dot(const MatrixBase<DerivedA>& v1, const MatrixBase<DerivedB>& v2)
{ return v1.dot(v2); }

template<typename VectorType>
inline typename VectorType::Scalar
mag(const VectorType& v) { return v.norm(); }

template<typename VectorType>
inline typename VectorType::Scalar
mag2(const VectorType& v) { return v.squaredNorm(); }

template<typename VectorType>
inline void
normalize(VectorType& v) { v.normalize(); }

template<typename DerivedA, typename DerivedB>
inline typename DerivedA::Scalar
cross(const MatrixBase<DerivedA>& v1, const MatrixBase<DerivedB>& v2)
{ return v1[0]*v2[1] - v1[1]*v2[0]; }

template<typename T>
T inf() { return std::numeric_limits<T>::infinity(); }

// returns alternative system A' and b' with zero pivots removed and a map to
// the original system
template<typename T>
std::tuple<Eigen::SparseMatrix<T, Eigen::RowMajor>, Eigen::Matrix<T, Eigen::Dynamic, 1>, std::vector<int>>
remove_zero_pivots_row_major(const Eigen::SparseMatrix<T,Eigen::RowMajor>& A,
                             const Eigen::Matrix<T, Eigen::Dynamic, 1>& b)
{
  std::vector<int> to_original;
  std::vector<int> to_new(A.rows(), -1);

  for (int k = 0; k < A.outerSize(); ++k)
  {
    typename Eigen::SparseMatrix<T,Eigen::RowMajor>::InnerIterator it(A,k);
    if ( it ) 
    {
      to_new[k] = to_original.size();
      to_original.push_back(k);
    }
  }

  auto new_size = to_original.size();
  Eigen::Matrix<T, Eigen::Dynamic, 1> newb(new_size);

  std::vector<Triplet<T>> triplets;
  for ( int row = 0; row < new_size; ++row )
  {
    auto k = to_original[row];
    newb[row] = b[k];
    // collect non-zeros for the new matrix
    for (typename Eigen::SparseMatrix<T,Eigen::RowMajor>::InnerIterator it(A,k); it; ++it)
    {
      assert( to_new[it.col()] != -1 );
      triplets.push_back(Triplet<T>(row, to_new[it.col()], it.value()));
    }
  }
  Eigen::SparseMatrix<T,Eigen::RowMajor> newA(new_size,new_size);
  newA.setFromTriplets(triplets.begin(), triplets.end());

  return std::make_tuple(std::move(newA), std::move(newb), std::move(to_original));
}

template<typename T>
void
remove_zero_pivots_col_major(const Eigen::SparseMatrix<T>& A,
                             const Eigen::Matrix<T, Eigen::Dynamic, 1>& b,
                             Eigen::SparseMatrix<T>& newA,
                             Eigen::Matrix<T, Eigen::Dynamic, 1>& newb,
                             std::vector<int>& to_original)
{
  std::vector<int> to_new(A.cols(), -1);

  for (int k = 0; k < A.outerSize(); ++k)
  {
    typename Eigen::SparseMatrix<T>::InnerIterator it(A,k);
    if ( it ) 
    {
      to_new[k] = to_original.size();
      to_original.push_back(k);
    }
  }

  auto new_size = to_original.size();
  newb.resize(new_size);

  std::vector<Triplet<T>> triplets;
  for ( int col = 0; col < new_size; ++col )
  {
    auto k = to_original[col];
    newb[col] = b[k];
    // collect non-zeros for the new matrix
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(A,k); it; ++it)
    {
      assert(to_new[it.row()] != -1);
      triplets.push_back( Triplet<T>(to_new[it.row()], col, it.value()) );
    }
  }
  newA.resize(new_size,new_size);
  newA.reserve(triplets.size());
  newA.setFromTriplets(triplets.begin(), triplets.end());
}

template<typename T>
void write_julia(std::ostream &out, const Eigen::SparseMatrix<T>& A, const Eigen::Matrix<T, Eigen::Dynamic, 1>& b)
{
  out << "A = sparse([";
  for (int k = 0; k < A.outerSize(); ++k)
  {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(A,k); it; ++it)
    {
      out << it.row() + 1 << ",";
    }
  }
  out << "],\n [";
  for (int k = 0; k < A.outerSize(); ++k)
  {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(A,k); it; ++it)
    {
      out << it.col() + 1 << ",";
    }
  }
  out << "],\n [";

  for (int k = 0; k < A.outerSize(); ++k)
  {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it(A,k); it; ++it)
    {
      out << it.value() << ",";
    }
  }
  out << "], " << A.innerSize() << ", " << A.outerSize() << ")\n";

  out << "b = [";
  for ( int k = 0; k < b.size(); ++k )
  {
    out << b[k] << ",";
  }
  out << "]" << std::endl;
}
