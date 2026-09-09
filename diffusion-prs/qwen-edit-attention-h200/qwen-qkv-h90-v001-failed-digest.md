Vector-cache v1 correctness failure:2failed/8passed, includingactual8152/1365shape anddensemodel-forward; no microbenchmark orNCU executed.8ca388bde9 failedversion preserved.

Offlinecuobjdump ofthelatest compiledmodule (pathandSASSsavedhere) confirms theintendedtwo64bitcacheloads at0950/0960. The real RoPE component stillusesfma(x,cos,-round(y*sin)), buttheimaginarycomponent changedfromparentfma(y,cos,round(x*sin)) tonewfma(x,sin,round(y*cos)). The compiler changedwhich productwasroundedwhenbothcachevaluesbecameavailable together. Sum-of-squares andnormalizationmul orderingwerecheckedandremainconsistent. This is a generated-code difference, notanallowable numericaltolerance change.

Nextedit: express theexistingparentcontraction with __fmaf_rn and __fmul_rn forbothRoPEcomponents, thenrequireall10exacttests beforebenchmarking/profilingagain.
