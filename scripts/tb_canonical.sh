cd $OPENAI_LOGDIR
$ACTIVATE_GYM_PYTHON
tensorboard --logdir=\
v6:pyERSEnv-ca-dynamic-30-v6,\
blipsv6:pyERSEnv-ca-dynamic-blips-30-v6,\
sgv6:SgERSEnv-ca-dynamic-30-v6,\
sgblipsv6:SgERSEnv-ca-dynamic-blips-30-v6,\
capv6:pyERSEnv-ca-dynamic-cap8-30-v6,\
blipscapv6:pyERSEnv-ca-dynamic-blips-cap8-30-v6,\
sgcapv6:SgERSEnv-ca-dynamic-cap8-30-v6,\
sgblipscapv6:SgERSEnv-ca-dynamic-blips-cap8-30-v6,\
nc1v6:pyERSEnv-ca-dynamic-constraints1-30-v6,\
blipsnc1v6:pyERSEnv-ca-dynamic-blips-constraints1-30-v6,\
sgnc1v6:SgERSEnv-ca-dynamic-constraints1-30-v6, \
sgblipsnc1v6:SgERSEnv-ca-dynamic-blips-constraints1-30-v6, \
bssv0:BSSEnv-v0,\
bssv1:BSSEnv-v1,\
bssv2:BSSEnv-v2\
 $1 $2 $3 $4 $5
