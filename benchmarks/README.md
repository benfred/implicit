Benchmarks
==========

All benchmarks reported here were performed on an Intel i7-7820x CPU. GPU Benchmarks were done
on a NVIDIA A6000.

### Spark Comparison

The benchmark_spark.py script compares the AlternatingLeastSquares model found here
to the implementation found in [Spark MLlib](https://spark.apache.org/mllib/).

To run this comparison, you should first [compile Spark with native BLAS
support](https://github.com/Mega-DatA-Lab/SpectralLDA-Spark/wiki/Compile-Spark-with-Native-BLAS-LAPACK-Support
).

This benchmark compares the Conjugate Gradient solver found in implicit on both the CPU and GPU,
to the Cholesky solver used in Spark.

The times per iteration are average times over 5 iterations.

#### last.fm 360k dataset

For the lastm.fm dataset at 256 factors, implicit on the CPU is 30x faster than Spark and the GPU version of implicit is
 260x faster than Spark:

![last.fm als train time](./spark_speed_lastfm.png)

<!--
{'Implicit (GPU)': {64: 0.44644722938537595,
  128: 0.38864846229553224,
  192: 0.5134327411651611,
  256: 0.6314576625823974},
'Implicit (CPU)': {64: 2.3200541973114013,
  128: 2.8571616649627685,
  192: 3.833038663864136,
  256: 5.307447624206543},
 'Spark MLlib': {64: 43.841095733642575,
  128: 65.43235535621643,
  192: 104.12412366867065,
  256: 164.2230523586273}}
-->


#### MovieLens 20M dataset

For the ml20m dataset at 256 factors, implicit on the CPU was 23x faster than Spark while the GPU version
was 180x faster than Spark:

![als train time](./spark_speed_ml20m.png)
<!--
{'Implicit (GPU)': {64: 0.3278830528259277,
  128: 0.22653441429138182,
  192: 0.26464319229125977,
  256: 0.29997830390930175},
'Implicit (CPU)': {64: 0.8396152973175048,
  128: 1.2089608192443848,
  192: 1.6725208759307861,
  256: 2.3451559066772463},
 'Spark MLlib': {64: 12.285535764694213,
  128: 19.666392993927,
  192: 33.25573806762695,
  256: 54.00092940330505}}
-->

Note that this dataset was filtered down for all versions to reviews that were positive (4+
stars), to simulate a truly implicit dataset.
