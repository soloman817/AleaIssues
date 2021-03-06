﻿module Sample.XorShift7.F2

type Vector32(bits:uint32) =
    let mutable bits = bits

    new () = Vector32(0u)

    member this.Bits with get () = bits and set v = bits <- v

    static member ( * ) (lhs:Vector32, rhs:Vector32) =
        let mutable r = 0u
        let p = lhs.Bits &&& rhs.Bits
        r <- p
        r <- (r >>> 16) ^^^ (r &&& 0xffffu)
        r <- (r >>> 8) ^^^ (r &&& 0xffu)
        r <- (r >>> 4) ^^^ (r &&& 0xfu)
        r <- (r >>> 2) ^^^ (r &&& 0x3u)
        r <- (r >>> 1) ^^^ (r &&& 0x1u)
        r

type Vector256(bits:uint32[]) =

    let bits = Array.copy bits

    new () = Vector256(Array.zeroCreate 8)

    member this.Bits = bits

    static member ( * ) (lhs:Vector256, rhs:Vector256) =
        let mutable r = 0u
        let mutable p = 0u
        for i = 0 to 7 do
            p <- p ^^^ (lhs.Bits.[i] &&& rhs.Bits.[i])
        r <- p
        r <- (r >>> 16) ^^^ (r &&& 0xffffu)
        r <- (r >>> 8) ^^^ (r &&& 0xffu)
        r <- (r >>> 4) ^^^ (r &&& 0xfu)
        r <- (r >>> 2) ^^^ (r &&& 0x3u)
        r <- (r >>> 1) ^^^ (r &&& 0x1u)
        r

type Matrix32(bits:uint32[]) =

    let bits = Array.copy bits
    
    new() = Matrix32(Array.zeroCreate 32)

    member this.Bits = bits

    static member Identity =
        let bits = Array.zeroCreate 32
        let mutable value = 1u <<< 31
        for i = 0 to 31 do
            bits.[i] <- value
            value <- value >>> 1
        Matrix32(bits)

    static member Left(n:int) =
        if not (n >= 1 && n <= 31) then failwith "error"
        let bits = Array.zeroCreate 32
        let mutable value = 1u <<< (31 - n)
        for i = 0 to 31 do
            bits.[i] <- value
            value <- value >>> 1
        Matrix32(bits)

    static member Right(n:int) = 
        if not (n >= 1 && n <= 31) then failwith "error"
        let bits = Array.zeroCreate 32
        let mutable value = 1u <<< n
        let mutable i = 31
        while i >= 0 do
            bits.[i] <- value
            value <- value <<< 1
            i <- i - 1
        Matrix32(bits)

    static member ( + ) (lhs:Matrix32, rhs:Matrix32) = 
        let r = Matrix32()
        for i = 0 to 31 do
            r.Bits.[i] <- lhs.Bits.[i] ^^^ rhs.Bits.[i]
        r

    static member ( * ) (lhs:Matrix32, rhs:Matrix32) =
        let r = Matrix32()
        for i = 0 to 31 do
            let c = Vector32()
            for j = 0 to 31 do
                c.Bits <- c.Bits ||| (((rhs.Bits.[j] >>> (31 - i)) &&& 0x1u) <<< (31 - j))
            for j = 0 to 31 do
                r.Bits.[j] <- r.Bits.[j] ||| ((Vector32(lhs.Bits.[j]) * c) <<< (31 - i))
        r

type Matrix256(bits:uint32[][]) =

    let bits = Array.init 256 (fun i -> Array.copy bits.[i])

    new () = Matrix256(Array.init 256 (fun _ -> Array.zeroCreate 8))

    member this.Bits = bits

    static member ( * ) (lhs:Matrix256, rhs:Matrix256) =
        let r = Matrix256()
        for i = 0 to 7 do
            for j = 0 to 31 do
                let bits = Array.zeroCreate 8
                for k = 0 to 7 do
                    for l = 0 to 31 do
                        bits.[k] <- bits.[k] ||| (((rhs.Bits.[k * 32 + l].[i] >>> (31 - j)) &&& 0x1u) <<< (31 - l))
                let c = Vector256(bits)
                for k = 0 to 255 do
                    r.Bits.[k].[i] <- r.Bits.[k].[i] ||| ((Vector256(lhs.Bits.[k]) * c) <<< (31 - j))
        r

    static member ( * ) (lhs:Matrix256, rhs:Vector256) =
        let r = Vector256()
        for i = 0 to 7 do
            for j = 0 to 31 do
                let t = Vector256(lhs.Bits.[i * 32 + j])
                r.Bits.[i] <- r.Bits.[i] ||| ((t * rhs) <<< (31 - j))
        r

    member this.Set32x32Block(row:int, col:int, b:Matrix32) =
        if not (row >= 0 && row < 8) then failwith "error"
        if not (col >= 0 && col < 8) then failwith "error"

        for i = 0 to 31 do
            bits.[32 * row + i].[col] <- b.Bits.[i]

    member this.PowPow2(p:int) =
        if not (p >= 0) then failwith "error"
        let mutable r = Matrix256()
        for i = 0 to 255 do
            for j = 0 to 7 do
                r.Bits.[i].[j] <- bits.[i].[j]
        for i = 0 to p - 1 do
            r <- r * r
        r

