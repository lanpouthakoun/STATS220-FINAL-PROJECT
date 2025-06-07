#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _capump_reg(void);
extern void _ingauss_reg(void);
extern void _mammalian_spike_35_reg(void);
extern void _mammalian_spike_reg(void);
extern void _xtra_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"../nrn/capump.mod\"");
    fprintf(stderr, " \"../nrn/ingauss.mod\"");
    fprintf(stderr, " \"../nrn/mammalian_spike_35.mod\"");
    fprintf(stderr, " \"../nrn/mammalian_spike.mod\"");
    fprintf(stderr, " \"../nrn/xtra.mod\"");
    fprintf(stderr, "\n");
  }
  _capump_reg();
  _ingauss_reg();
  _mammalian_spike_35_reg();
  _mammalian_spike_reg();
  _xtra_reg();
}

#if defined(__cplusplus)
}
#endif
