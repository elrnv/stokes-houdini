#pragma once

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

namespace sim_stokes_options
{
  // Scheme used in computing the stokes equations
  enum Scheme
  {
    STOKES,
    DECOUPLED,
    DECOUPLED_FANCY,
    DECOUPLED_NOEXPANSION,
    DECOUPLED_NOEXPANSION_FANCY,
    PRESSURE_ONLY,
    VISCOSITY_ONLY
  };

  enum FloatPrecision
  {
    FLOAT32,
    FLOAT64
  };
}

class SIM_Stokes : public GAS_SubSolver
{
public:
  /// These macros are used to create the accessors
  /// getFieldDstName and getFieldSrcName functions we'll use
  /// to access our data options.
  GET_DATA_FUNC_F(SIM_NAME_SCALE, Scale);
  GET_DATA_FUNC_E("scheme", Scheme, sim_stokes_options::Scheme);
  GET_DATA_FUNC_I("numsupersamples", NumSuperSamples);
  GET_DATA_FUNC_E("floatprecision", FloatPrecision, sim_stokes_options::FloatPrecision);
  GET_DATA_FUNC_B(SIM_NAME_OPENCL, UseOpenCL);
  GET_DATA_FUNC_F(SIM_NAME_TOLERANCE, Tolerance);
  GET_DATA_FUNC_F("maxdensity", MaxDensity);
  GET_DATA_FUNC_F("mindensity", MinDensity);
  GET_DATA_FUNC_F("minviscosity", MinViscosity);

protected:
  explicit  SIM_Stokes(const SIM_DataFactory *factory);
  virtual  ~SIM_Stokes();

  /// Used to determine if the field is complicated enough to justify
  /// the overhead of multithreading.
  bool shouldMultiThread(const SIM_RawField *field) const 
  { return field->field()->numTiles() > 1; }

  /// The overloaded callback that GAS_SubSolver will invoke to
  /// perform our actual computation.  We are giving a single object
  /// at a time to work on.
  virtual bool solveGasSubclass(
      SIM_Engine &engine,
      SIM_Object *obj,
      SIM_Time time,
      SIM_Time timestep);

private:
  /// We define this to be a DOP_Auto node which means we do not
  /// need to implement a DOP_Node derivative for this data.  Instead,
  /// this description is used to define the interface.
  static const SIM_DopDescription     *getDopDescription();

  /// These macros are necessary to bind our node to the factory and
  /// ensure useful constants like BaseClass are defined.
  DECLARE_STANDARD_GETCASTTOTYPE();
  DECLARE_DATAFACTORY(SIM_Stokes,
      GAS_SubSolver,
      "Stokes",
      getDopDescription());
};
