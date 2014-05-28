module Sample.XorShift7.Test

open System
open System.IO
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Utilities
open NUnit.Framework
open Sample.XorShift7
open Sample.XorShift7.F2

let assertArrayEqual (eps:float option) (A:'T[]) (B:'T[]) =
    (A, B) ||> Array.iter2 (fun a b -> eps |> function
        | None -> Assert.AreEqual(a, b)
        | Some eps -> Assert.That(b, Is.EqualTo(a).Within(eps)))

let writeBitcode (bitcode:byte[]) (name:string) =
    let desktopFolder = Environment.GetFolderPath(Environment.SpecialFolder.Desktop)
    use file = new FileStream (Path.Combine(desktopFolder, name), FileMode.Create, FileAccess.ReadWrite, FileShare.ReadWrite)
    use mem = new MemoryStream (bitcode)
    mem.CopyTo file

let template (convertExpr:Expr<uint32 -> 'T>) = cuda {
    let! kernel = GPU.kernel convertExpr |> Compiler.DefineKernel

    return Entry(fun (program:Program) ->
        let worker = program.Worker
        let kernel = program.Apply(kernel)

        // Copy pre-calculated bit-matrices, needed for jump-ahead
        // calculations, to the device memory.
        let jumpAheadMatrices = worker.Malloc(Data.jumpAheadMatrices)

        let generate (streams:int) (steps:int) (seed:uint32) (runs:int) (rank:int) =
            // first create state0 and scatter into device
            let state0 = Common.generateStartState seed
            use state0 = worker.Malloc(state0)
            // then create random number memory
            use numbers = worker.Malloc<'T>(streams * steps)
            // calculate the launch param
            let lp = GPU.launchParam streams
            // just launch
            kernel.Launch lp runs rank state0.Ptr jumpAheadMatrices.Ptr steps numbers.Ptr
            numbers.Gather()

        generate ) }

let correctness (convertD:Expr<uint32 -> 'T>) (convertH:uint32 -> 'T) options =
    let worker = Worker.Default
    let suffix = 
        match options.LinkageOpt with
        | LinkageOpt.O0 -> "O0"
        | LinkageOpt.O3 -> "O3"

    let irm = Compiler.Compile(template convertD, options).IRModule
    writeBitcode irm.Bitcode (sprintf "irm_%s.ll" suffix)

    //let ptxm = Compiler.Link(irm, worker.Device.Arch).PTXModule
    let ptxm = Compiler.Link(irm).PTXModule
    writeBitcode ptxm.Bitcode (sprintf "ptxm_%s.ptx" suffix)

    use program = worker.LoadProgram(ptxm)

    let streams = 4096
    let steps = 5
    let seed = 42u
    let runs = 1
    let rank = 0

    let hOutputs =
        let result = Array.zeroCreate (streams * steps)
        let mutable p = 0
        while (1 <<< p) < streams do p <- p + 1
        let state = Common.generateStartState seed
        let m = CPU.XorShift7Rng.Matrix.PowPow2(256 - p)
        let mutable v = Vector256(state)
        for i = 0 to streams - 1 do
            let rng = CPU.XorShift7Rng(v.Bits)
            for j = 0 to steps - 1 do
                let number = rng.NextUniformUInt32() |> convertH
                result.[j * streams + i] <- number
            v <- m * v
        result

    let dOutputs = program.Run streams steps seed runs rank

    printfn "%A" hOutputs
    printfn "%A" dOutputs

    assertArrayEqual None hOutputs dOutputs

let options1 = { CompileOptions.ProfilingConfig with LinkageOpt = LinkageOpt.O0 }
let options2 = { CompileOptions.ProfilingConfig with LinkageOpt = LinkageOpt.O3 }
let test options = correctness <@ id @> id options
let [<Test>] testO0() = test options1
let [<Test>] testO3() = test options2
