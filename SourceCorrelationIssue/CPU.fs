module Sample.XorShift7.CPU

open Alea.CUDA.Utilities
open Sample.XorShift7
open Sample.XorShift7.F2

(*
type XorShift7Rng(state:uint32[]) =
    let state = Array.copy state
    let mutable index = 0

    new () = XorShift7Rng(Array.zeroCreate 8)

    new (seed:uint32) = XorShift7Rng(Common.generateStartState(seed))

    member this.State = state

    static member Matrix =
        let r = Matrix256()
        let zero = Matrix32()
        let identity = Matrix32.Identity
        for row = 0 to 6 do
            for col = 0 to 7 do
                r.Set32x32Block(row, col, if row = col - 1 then identity else zero)
        r.Set32x32Block(7, 0, (identity + Matrix32.Left(24)) * (identity + Matrix32.Right(7)))
        r.Set32x32Block(7, 1, identity + Matrix32.Right(10))
        r.Set32x32Block(7, 2, zero)
        r.Set32x32Block(7, 3, identity + Matrix32.Right(3))
        r.Set32x32Block(7, 4, identity + Matrix32.Left(7))
        r.Set32x32Block(7, 5, zero)
        r.Set32x32Block(7, 6, zero)
        r.Set32x32Block(7, 7, (identity + Matrix32.Left(9)) * (identity + Matrix32.Left(13)))
        r

    member this.NextUniformUInt32() =
        let mutable r = 0u
        let mutable t = 0u
        t <- state.[(index + 7) &&& 0x7]
        t <- t ^^^ (t <<< 13)
        r <- t ^^^ (t <<< 9)
        t <- state.[(index + 4) &&& 0x7]
        r <- r ^^^ (t ^^^ (t <<< 7))
        t <- state.[(index + 3) &&& 0x7]
        r <- r ^^^ (t ^^^ (t >>> 3))
        t <- state.[(index + 1) &&& 0x7]
        r <- r ^^^ (t ^^^ (t >>> 10))
        t <- state.[index]
        t <- t ^^^ (t >>> 7)
        r <- r ^^^ (t ^^^ (t <<< 24))
        state.[index] <- r
        index <- (index + 1) &&& 0x7
        r

    member this.NextUniformFloat32() =
        this.NextUniformUInt32() |> Common.toFloat32

    member this.NextUniformFloat64() =
        this.NextUniformUInt32() |> Common.toFloat64
*)

/// XorShift7 represents the implementation of xorshift7 RNG, as described in
/// http://www.iro.umontreal.ca/~lecuyer/myftp/papers/xorshift.pdf.
type XorShift7Rng(state : uint32[], i : int) =      

    /// The mutable state of the generator
    let state = Array.copy state

    /// The index of the generator in the state vector
    let mutable index = i
        
    /// Optimized jump ahead calculation based on precalculated matrix powers.
    ///
    /// Within numStreams streams jump ahead to stream streamId based on 
    /// initial state vector startState
    ///
    /// Calculate first power of 2 greater or equal to the number of
    /// threads; if that number is p, than each thread should start
    /// with the state that is 2^(256-p) steps distant from states of
    /// its neighboring (by rank) threads. Jumping ahead to the
    /// corresponding state could be accomplished by multiplying
    /// initial RNG state, interpreted as lenght-256 bit-vector, by
    /// corresponding update matrix.  If xorshift7 RNG specific
    /// 256x256 state update bit-matrix is denoted by M, and initial
    /// state bit-vector by v, then initial state of the thread with
    /// rank 0 could be calculated as (M^(2^(256-p)))^0*v, initial
    /// state of thread with rank 1 as (M^(2^(256-p)))^1*v, initial
    /// state of thread with rank 2 as (M^(2^(256-p)))^2*v, etc.
    /// Thus, matrix M^(2^(256-p)) appears as base matrix for these
    /// calculations. Matrices M^(2^224), M^(2^225) etc. up to
    /// M^(2^255) are pre-calculated and, having in mind that number
    /// of threads is integer number, thus value of p is less or
    /// equal to 31, matrix M^(2^(256-p)) (let's denote it B) is
    /// among these, so the pointer to this matrix is initialized
    /// here to point to corresponding matrix in the sequence of
    /// pre-calculated matrices supplied as argument to the kernel.
    ///
    /// Jump ahead, according to the thread rank, to RNG state
    /// appopriate for given thread; if thread rank is denoted by r,
    /// then jumping ahead could be accomplished through multiplying
    /// start state by matrix B raised to the thread rank: B^r*v.
    /// Thread rank could be interpreted as base-2 number
    /// b[p-1]...b[1]b[0] (where b[0] is the least significant bit,
    /// etc.), and then state update could be interpreted as follows:
    ///   B^r*v=B^(b[p-1]*2^(p-1)+...+b[1]*2^1+b[0]*2^0)*v=
    ///        =B^(b[p-1]*2^(p-1))*...*B^(b[1]*2^1)*B^(b[0]*2^0)*v
    ///        =(B^(2^(p-1)))^b[p-1]*...*(B^(2^1)^b[1]*(B^(2^0))^b[0]*v
    /// All of matrices (B^*(2^0)), (B^(2^1)), ..., B^(2^(p-1)) are
    /// pre-calculated, so the above expression could be calculated in
    /// a loop, from right-to-left.
    ///
    /// Note that we cannot take default constructed stream XorShift7() because this
    /// has a state with all elements equal to zero and therefore the 
    /// jump ahead states are all equal to zero as well
    /// however, using a seed of 0 works because of the way the generator is seeded
    static let jumpAheadMatrixSize = 256 * 8
    static member JumpAhead numStreams streamId (startState : uint32[]) =
        if numStreams < 0 then failwith "number of streams must be > 0"
        if streamId < 0 || streamId >= numStreams then failwith "partition id must be in [0,%d)" numStreams 
        if startState.Length <> 8 then failwith "initial state must be a vector of unsigned ints of length 8"
        if startState = Array.zeroCreate 8 then failwith "initial state cannot be all zero"

        // calculate 2^p such that 2^p >= numPartitions
        let mutable p = 0
        while (1 <<< p) < numStreams do p <- p + 1

        // id of matrix B injumpAheadMatrices
        let jumpAheadMatrices = Data.jumpAheadMatrices
        let jumpAheadMatrixIdx = (32 - p) * jumpAheadMatrixSize 

        // multiply with matrix (B^(2^(i-1))) if bit i is set in streamId
        let updateState (i:int) (state:Vector256) =
            if streamId &&& (1 <<< i) <> 0 then
                let offset = jumpAheadMatrixIdx + i*jumpAheadMatrixSize
                let bits = Array.init 256 (fun i -> Array.init 8 (fun j -> jumpAheadMatrices.[offset + i*8 + j])) 
                let B = Matrix256(bits)  
                B * state 
            else
                state

        let startState = Vector256(startState)
        let finalState = seq {0..p-1} |> Seq.fold (fun state i -> updateState i state) startState            
        finalState.Bits       
            
    /// Default class constructor, creates RNG with all state bits set to 0.
    new() = XorShift7Rng(Array.zeroCreate 8, 0)
       
    /// Alternative constructor, creates xorshift RNG with state
    /// initialized from sequence of random numbers, generated using
    /// linear congruential RNG, and starting with the given seed.
    /// @param seed the seed for linear congruential RNG, used to
    /// produce an array of 8 unsigned numbers representing initial
    /// state of the xorhsift7 RNG      
    new(seed : uint32) = XorShift7Rng(Common.generateStartState(seed), 0) 

    /// Constructor with jump ahead 
    /// Constructing generator with index streamId out of numStreams generators
    /// To run the generator with say 4 streams use the following code
    ///     let rng1 = XorShift7(4, 0, 0u)
    ///     let rng2 = XorShift7(4, 1, 0u)
    ///     let rng3 = XorShift7(4, 2, 0u)
    ///     let rng4 = XorShift7(4, 3, 0u)
    // This will produce 4 independent subsequences of the whole period
    new(numStreams : int, streamId : int, seed : uint32) =
        let startState = Common.generateStartState seed
        let jumpAheadState = XorShift7Rng.JumpAhead numStreams streamId startState
        XorShift7Rng(jumpAheadState, 0) 

    /// Constructor with jump ahead and initial state
    new(numStreams : int, streamId : int, startState : uint32[]) =
        let jumpAheadState = XorShift7Rng.JumpAhead numStreams streamId startState
        XorShift7Rng(jumpAheadState, 0) 

    /// Another alternative constructor, creates xorshift RNG with
    /// state set to values from given array of numbers (and index to
    /// last number set to 0).
    /// @param state pointer to an array of 8 unsigned numbers
    /// representing values to initialize RNG state with     
    new(s : uint32[]) = XorShift7Rng(s, 0) 

    /// The xorshift7 RNG state is represented by 8 unsigned 32-bit numbers
    member this.State = state

    /// The array of numbers used for xorshift7 state is used as
    /// circular buffer, with following member variable pointing to
    /// last (next to be reused) number in the buffer. 
    member this.Index = index

    /// The state updates of xorshift7 RNG could be calculated by
    /// multiplying current state (imagined as length-256 bit-vector,
    /// with least significant 32 bits represented by last number in
    /// the state buffer, next 32 bits represented by next-to-last
    /// number in the buffer, etc.) with specific 256x256 matrix, and
    /// this function is building and returning this matrix.  As
    /// given matrix is rather sparse, state is updated much faster
    /// through direct calculations, implemented by
    /// Xorshift7::getUniform() method; however, this state update
    /// matrix is usable for jumping ahead RNG for large number of steps.
    /// Returns xorshift7 RNG state update matrix
    static member Matrix =
        let r = Matrix256()
        let zero = Matrix32()
        let identity = Matrix32.Identity
        for row = 0 to 6 do
            for col = 0 to 7 do
                r.Set32x32Block(row, col, if row = col - 1 then identity else zero)
        r.Set32x32Block(7, 0, (identity + Matrix32.Left(24)) * (identity + Matrix32.Right(7)))
        r.Set32x32Block(7, 1, identity + Matrix32.Right(10))
        r.Set32x32Block(7, 2, zero)
        r.Set32x32Block(7, 3, identity + Matrix32.Right(3))
        r.Set32x32Block(7, 4, identity + Matrix32.Left(7))
        r.Set32x32Block(7, 5, zero)
        r.Set32x32Block(7, 6, zero)
        r.Set32x32Block(7, 7, (identity + Matrix32.Left(9)) * (identity + Matrix32.Left(13)))
        r
        
    /// Get numbers representing xorshift7 RNG state to given array.
    member this.GetState() =
        Array.init 8 (fun i -> state.[(index + i) &&& 0x7])
                             
    /// Produce next random number from uniform distribution, as an
    /// unsigned number from the range of all 32-bit unsigned
    /// numbers, and update RNG state
    member this.NextUniformUInt32() =
        // calculate next random number, and update xorshift7 RNG state,
        // again according to the RNG definition as given by above mentioned paper
        let mutable t = 0u
        let mutable r = 0u
        t <- state.[(index + 7) &&& 0x7];
        t <- t ^^^ (t <<< 13)
        r <- t ^^^ (t <<< 9)
        t <- state.[(index + 4) &&& 0x7]
        r <- r ^^^ (t ^^^ (t <<< 7))
        t <- state.[(index + 3) &&& 0x7]
        r <- r ^^^ (t ^^^ (t >>> 3))
        t <- state.[(index + 1) &&& 0x7]
        r <- r ^^^ (t ^^^ (t >>> 10))
        t <- state.[index]
        t <- t ^^^ (t >>> 7)
        r <- r ^^^ (t ^^^ (t <<< 24))
        state.[index] <- r
        index <- (index + 1) &&& 0x7
        r          

let inline uniform (real:RealTraits<'T>) (generator:XorShift7Rng) (numbers:'T[]) =
    let norm = __oneover2to32minus1()
    let n = numbers.Length
    for i = 0 to n - 1 do
        let p = generator.NextUniformUInt32() |> real.Of
        numbers.[i] <- norm * p

let inline exponential (real:RealTraits<'T>) (generator:XorShift7Rng) (numbers:'T[]) =
    let norm = __oneover2to32minus1
    let n = numbers.Length
    for i = 0 to n - 1 do
        let p = generator.NextUniformUInt32() |> real.Of
        numbers.[i] <- log(norm * p)

let inline normal (real:RealTraits<'T>) (generator:XorShift7Rng) (numbers:'T[]) =
    let norm = __oneover2to32minus1()
    let pi = __pi()
    let n = numbers.Length
    let mutable x2 = 0G
    for i = 0 to n - 1 do
        let p = generator.NextUniformUInt32() |> real.Of
        if i % 2 = 0 then
            x2 <- sqrt((real.Of -2) * log(norm * p))
        else
            numbers.[i - 1] <- x2 * sin(2G * pi * norm * p)
            if i < n then
                numbers.[i] <- x2 * cos(2G * pi * norm * p)

