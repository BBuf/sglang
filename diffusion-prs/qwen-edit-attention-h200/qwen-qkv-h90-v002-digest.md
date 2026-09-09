### NCU Report Digest: vector-cache Qwen epilogue @6c1d2cf1a8

Parent:../qwen-qkv-h90-v000,38ee. SameH200shape8152/1365/H24D128BF16 andfocusedharness. All10exacttests and6productionmicrocasespassed. NCUclock/cachecontrolnone;report containsvariable-clockwarning,so smalltimedeltasarenotacceptedE2Eproof. CUDAlineinfo/PTXunavailable;SASS/sourcecounterCSV available. PMsectionscaptured butsomePMrawcolumns show"no data";useaggregatePCsamples only.

Measured: parent121.248us→v2 121.760us (noimprovement);independenttinyTorchtrace121.889→119.520us (smalldifferencewithinmeasurementvariance). Excesssectors7,309,056→0 andtotaltheoreticalsectors29,236,224→21,927,168;idealunchanged21,927,168. Executedinstructions77,148,648→74,847,672. SASS confirms2LDG64cacheloads. DRAM61.54%,L269.94%,SM61.46%,occupancy89.878%,eligiblewarps2.161. Coalescingchangedthepredictedcountersbutnotmainlatency.

V-copySTG at0x7b3fa77cd270:3,782/14,291allPCsamples (26.46%),including3,739long-scoreboardsamples and1,938not-issuedlongscoreboard samples. It directlydependsonthe64bitinputLDG withoutindependentwork. Q/Kinputconversionalsowaitsonloads(2,682longscoreboardsamples),butVcopyisthenextboundedtarget. This ismemorylatency/insufficientindependentbytesperwarp,notspillorbranchdivergence. Data-volume lowerbound remainsimportant;noneofthepotentialgainisclaimedyet.

Next Concrete Edit
- File:qwen_qkv_epilogue.cuh
- Change:compressVworkitems soeachwarp copies2adjacentheads viaone128bitper-lane load/store; retainallwarpsworkinganduseoriginal64bitcopyforanoddlasthead. Do not justskipoddheadsintheoldmapping,whichwouldhalveactiveVwarps.
- Validation:exacttestsincludingheads1/3/4/24andmixedtokenstrides;productionmicro;matchingNCU/SASS beforeanotherfullmodelABBA.
- Expectedmovement:Vload/storeinstructioncountabout½,unchangedbytes,lowerVlongscoreboardcycles andshorterkernelwithoutregister/occupancyregression.
