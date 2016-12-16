#include <UT/UT_DSOVersion.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include "solver/eigen.h"
#include "SIM/SIM_Stokes.h"
#include "SOP/SOP_Stokes.h"

///
/// This is the hook that Houdini grabs from the dll to link in
/// this.  As such, it merely has to implement the data factory
/// for this node.
///
void
initializeSIM(void *)
{
  Eigen::setNbThreads(12);
  if ( Eigen::nbThreads() > 1 )
    std::cout << "Eigen is multithreaded" << std::endl;
  IMPLEMENT_DATAFACTORY(SIM_Stokes);
}

// Register sop operator
#if 0
void
newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(new OP_Operator(
        "hdk_stokes",                   // Internal name
        "Stokes",                       // UI name
        SOP_Stokes::myConstructor,      // How to build the SOP
        SOP_Stokes::buildTemplates(),   // My parameters
        1,                              // Min # of sources
        2,                              // Max # of sources
        0,//SOP_Stokes::myVariables,    // Local variables
        OP_FLAG_GENERATOR));            // Flag it as generator
}
#endif

