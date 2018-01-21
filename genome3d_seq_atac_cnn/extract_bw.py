from genomedatalayer.shm import extract_bigwig_to_npy

import os
import sys

def makedirs(path, mode=0777, exist_ok=False):
    try:
        os.makedirs(path, mode)
    except OSError:
        if not exist_ok or not os.path.isdir(path):
            raise
        else:
            sys.stderr.write('Warning: directory %s exists.\n' % path)


makedirs("/srv/scratch/mtaranov/atac_pval_D0")
makedirs("/srv/scratch/mtaranov/atac_fc_D0")

extract_bigwig_to_npy('/srv/www/kundaje/dskim89/ggr/portal/2016_AUG/atac/signal/timepoint/pval/primary_keratinocyte-d00.GGR.Stanford_Greenleaf.ATAC-seq.b1.trim.PE2SE.nodup.tn5_pooled.pf.pval.signal.bigwig', '/srv/scratch/mtaranov/atac_pval_D0')
extract_bigwig_to_npy('/srv/www/kundaje/dskim89/ggr/portal/2016_AUG/atac/signal/timepoint/fc/primary_keratinocyte-d00.GGR.Stanford_Greenleaf.ATAC-seq.b1.trim.PE2SE.nodup.tn5_pooled.pf.fc.signal.bigwig', '/srv/scratch/mtaranov/atac_fc_D0')
