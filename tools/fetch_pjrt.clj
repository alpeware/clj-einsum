(ns tools.fetch-pjrt
  "Automated binary fetcher for precompiled PJRT shared objects from PyPI wheels."
  (:require [clojure.java.io :as io]
            [clojure.string :as str])
  (:import [java.io ByteArrayOutputStream]
           [java.nio.file Files Path]
           [java.nio.file.attribute FileAttribute]
           [java.util.zip ZipInputStream]))

(def SOURCES
  "Map of target backend to PyPI wheel download URL and extraction parameters."
  {:cpu
   {:url "https://files.pythonhosted.org/packages/5d/ca/5487881b30d07396c168749ead36f57dbc4538cafdc535943afc81b3e5a2/xla_cpu_pjrt-0.0.1-py3-none-manylinux_2_27_x86_64.whl"
    :so-name "libpjrt_cpu.so"
    :entry-pattern #"xla_cpu_pjrt\.so$"}

   :cuda12
   {:url "https://files.pythonhosted.org/packages/5f/78/a3d9ceda0793f4fb43daa292af7b801932611a1aed442636ddfc93d58c7a/jax_cuda12_pjrt-0.10.0-py3-none-manylinux_2_27_x86_64.whl"
    :so-name "libpjrt_cuda.so"
    :entry-pattern #"xla_cuda_plugin\.so$"}

   :sycl
   {:url "https://files.pythonhosted.org/packages/14/df/e2f2e519c22ed0d885a53e6b52a78e7808fa7a7b8e5d0f11d13f9f257a55/intel_extension_for_openxla-0.7.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    :so-name "libpjrt_sycl.so"
    :entry-pattern #"(xla_sycl_plugin|pjrt_sycl_plugin|pjrt_plugin_xpu)\.so$"
    :deps ["https://files.pythonhosted.org/packages/9a/1f/ff1b20fe45c1a2077f1a11e2447826881987a76983facab763a72ee29fc8/dpcpp_cpp_rt-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/91/91/3ea39f73fbacb8ee7d9779c7598275c695c784ae94cc77ea5c7f5ee47a4f/mkl_dpcpp-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/6d/b4/ef531295ed33b929c6c5214421eeebe370f1be22536b6956b4aaf18fdbc5/mkl-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/70/9a/a338de7fc24087c40d38401372618c8465103795baefe941eea3acf55678/intel_cmplr_lib_ur-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/3d/69/8050e96e5b099b349d9109f727ce39b4f57414a24d3eb71a12fdc48bad87/oneccl-2021.17.1-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/76/74/27684e2d0d32923da293f22b418c0c13919839444b67e86a8986f44938e7/intel_sycl_rt-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/3d/66/26dfd6a19f7faf595da12c21bdb4102c6f30755511c7f167f745b203fbb7/intel_cmplr_lib_rt-2025.3.1-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/27/8e/4a90b6aa955268988e7491f502b7ac2bd65cb954b4979bfcc892cf019b50/umf-1.0.2-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/1e/07/df0cd5b0ec5f0a0bcbc8e73e4b2cfca78449b3b521868b1e366bfe6f97a3/onemkl_sycl_blas-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/78/c4/c2cf3e1990707f7f1918f1073e3a26c56e92f06c1525af39501b271ede23/onemkl_sycl_lapack-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/a9/5f/4f0b81e5f83f5e42c549bd29a9398b507890ec24080d27cd7afada61521e/onemkl_sycl_sparse-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/7c/b8/1ec88922a9a479f567183cf82d49041451bcb7afbb3195202d9a57e5a0ff/onemkl_sycl_dft-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/c0/30/a5603eb7057a39e09d0debef5c6b84f2d04f30b7188014dffb06c59a0a23/onemkl_sycl_vm-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/94/6a/3fc34f47c69bdfbfce1f6d02f18e0fd41459bb4e1204e2a5cae179c0986e/onemkl_sycl_rng-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/6a/ce/f479c58454a11b7ac2d83231703b12064902604368e3a67e9b57fc4b9e72/onemkl_sycl_stats-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"
           "https://files.pythonhosted.org/packages/6f/2e/5a009451929da21a4898cb8f1925fc6a484e848cbed2e496aac5b14ab795/onemkl_sycl_datafitting-2025.3.0-py2.py3-none-manylinux_2_28_x86_64.whl"]}

   :rocm
   {:url "https://repo.amd.com/rocm/whl-multi-arch/jax_rocm7_plugin-0.10.0%2Brocm7.14.0-cp312-cp312-manylinux_2_27_x86_64.whl"
    :so-name "libpjrt_rocm.so"
    :entry-pattern #"(xla_rocm_plugin|pjrt_rocm_plugin)\.so$"
    :arch-pkgs ["https://archlinux.org/packages/extra/x86_64/rocprofiler/download/"
                "https://archlinux.org/packages/extra/x86_64/hsa-amd-aqlprofile/download/"]}})

(def URL-FALLBACKS
  "Map of primary download URLs to PyPI mirror URLs if primary yields HTTP 404."
  {"https://files.pythonhosted.org/packages/14/df/e2f2e519c22ed0d885a53e6b52a78e7808fa7a7b8e5d0f11d13f9f257a55/intel_extension_for_openxla-0.7.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
   "https://files.pythonhosted.org/packages/c6/2c/cf91c108d9480ef3f9866f7c0afbd567e25d9df75e90a5529ffa402f685a/intel_extension_for_openxla-0.7.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"})

(defn open-stream
  "Opens an input stream for `url`, falling back to `URL-FALLBACKS` if HTTP 404 occurs."
  [url]
  (try
    (io/input-stream url)
    (catch java.io.FileNotFoundException e
      (if-let [fallback (get URL-FALLBACKS url)]
        (do
          (println (str "Primary URL unavailable (" (.getMessage e) "), using mirror: " fallback))
          (io/input-stream fallback))
        (throw e)))))

(defn- extract-entry-bytes [zip]
  (let [baos (ByteArrayOutputStream.)]
    (io/copy zip baos)
    (.toByteArray baos)))

(defn- parse-symlink-target [^bytes bs]
  (when (< (alength bs) 300)
    (let [s (str/trim (String. bs "UTF-8"))]
      (when (re-find #"^[a-zA-Z0-9_.-]+$" (last (str/split s #"/")))
        s))))

(defn unpack-so-from-wheel
  "Streams zip archive from `url` and extracts the main `.so` matching `entry-pattern` to `out-file`."
  [url entry-pattern out-file]
  (with-open [in (open-stream url)
              zip (ZipInputStream. in)]
    (loop []
      (when-let [entry (.getNextEntry zip)]
        (if (re-find entry-pattern (.getName entry))
          (do
            (println (str "Extracting " (.getName entry) " -> " (.getAbsolutePath out-file)))
            (let [bs (extract-entry-bytes zip)]
              (if-let [link-target (parse-symlink-target bs)]
                (do
                  (Files/deleteIfExists (.toPath out-file))
                  (Files/createSymbolicLink (.toPath out-file) (Path/of link-target (into-array String [])) (make-array FileAttribute 0)))
                (do
                  (Files/deleteIfExists (.toPath out-file))
                  (io/copy bs out-file)))))
          (recur))))))

(defn unpack-all-sos-from-wheel
  "Streams zip archive from `url` and extracts ALL `.so` and `.so.*` files into `lib-dir` with symlink support."
  [url lib-dir]
  (.mkdirs lib-dir)
  (with-open [in (open-stream url)
              zip (ZipInputStream. in)]
    (loop []
      (when-let [entry (.getNextEntry zip)]
        (let [entry-name (.getName entry)
              file-name (last (str/split entry-name #"/"))]
          (when (and (not (.isDirectory entry))
                     (re-find #"\.so(\.\d+)*$" file-name))
            (let [target-file (io/file lib-dir file-name)
                  bs (extract-entry-bytes zip)]
              (if-let [link-target (parse-symlink-target bs)]
                ;; Symlink entry handling: create symlink if target-file is not already a real binary
                (when-not (and (.exists target-file) (not (Files/isSymbolicLink (.toPath target-file))))
                  (try
                    (Files/deleteIfExists (.toPath target-file))
                    (Files/createSymbolicLink (.toPath target-file) (Path/of link-target (into-array String [])) (make-array FileAttribute 0))
                    (catch Exception _
                      (io/copy bs target-file))))
                ;; Real binary payload extraction: overwrite symlinks or missing files
                (when (or (not (.exists target-file)) (Files/isSymbolicLink (.toPath target-file)))
                  (Files/deleteIfExists (.toPath target-file))
                  (io/copy bs target-file))))))
        (recur)))))

(defn fetch-binary
  "Fetches and unpacks native shared objects and dependencies for `target` into bin/ and bin/lib/."
  [target]
  (let [{:keys [url so-name entry-pattern deps arch-pkgs]} (get SOURCES target)
        _ (assert (some? url) (str "Unknown target: " target))
        bin-dir (io/file "bin")
        lib-dir (io/file bin-dir "lib")
        out-file (io/file bin-dir so-name)]
    (.mkdirs bin-dir)
    (println (str "Fetching " target " PJRT plugin from " url "..."))
    (unpack-so-from-wheel url entry-pattern out-file)
    ;; Extract all bundled .so dependencies from main wheel
    (unpack-all-sos-from-wheel url lib-dir)
    ;; Extract all .so dependencies from companion wheels
    (doseq [dep-url deps]
      (println (str "Fetching companion dependency wheel: " dep-url "..."))
      (unpack-all-sos-from-wheel dep-url lib-dir))
    ;; Extract companion Arch Linux packages if present
    (doseq [arch-url arch-pkgs]
      (println (str "Fetching companion Arch Linux package: " arch-url "..."))
      (try
        (let [cmd (str "curl -sL " arch-url " | unzstd | tar -C " (.getAbsolutePath lib-dir) " -xf - --strip-components=3 opt/rocm/lib/ 2>/dev/null || "
                       "curl -sL " arch-url " | unzstd | tar -C " (.getAbsolutePath lib-dir) " -xf - --strip-components=2 usr/lib/ 2>/dev/null || true")
              pb (ProcessBuilder. ["sh" "-c" cmd])
              proc (.start pb)]
          (.waitFor proc))
        (catch Exception e
          (println (str "Warning: Failed to extract Arch package (" (.getMessage e) ")")))))
    (when (= target :rocm)
      (let [counter-dir (io/file (.getParent lib-dir) "libexec/rocprofiler/counters")
            xml-file (io/file counter-dir "derived_counters.xml")]
        (.mkdirs counter-dir)
        (when-not (.exists xml-file)
          (.createNewFile xml-file))))
    ;; Ensure libsycl.so.9 and libccl.so.1 symlinks exist for SYCL backend
    (when (= target :sycl)
      (let [sycl8 (io/file lib-dir "libsycl.so.8")
            sycl9 (io/file lib-dir "libsycl.so.9")
            ccl2  (io/file lib-dir "libccl.so.2")
            ccl1  (io/file lib-dir "libccl.so.1")]
        (when (and (.exists sycl8) (not (.exists sycl9)))
          (println "Creating symlink libsycl.so.9 -> libsycl.so.8")
          (try
            (Files/createSymbolicLink (.toPath sycl9) (Path/of "libsycl.so.8" (into-array String [])) (make-array FileAttribute 0))
            (catch Exception _ nil)))
        (when (and (.exists ccl2) (not (.exists ccl1)))
          (println "Creating symlink libccl.so.1 -> libccl.so.2")
          (try
            (Files/createSymbolicLink (.toPath ccl1) (Path/of "libccl.so.2" (into-array String [])) (make-array FileAttribute 0))
            (catch Exception _ nil)))))
    (println (str "Successfully installed " (.getAbsolutePath out-file)))))

(defn -main
  [& args]
  (let [target-str (or (first args) "cpu")]
    (if (= target-str "all")
      (doseq [t (keys SOURCES)]
        (fetch-binary t))
      (fetch-binary (keyword target-str)))))

(when (= *file* (System/getProperty "clojure.script.filename"))
  (apply -main *command-line-args*))
