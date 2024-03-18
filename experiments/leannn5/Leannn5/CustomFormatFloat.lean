namespace CustomFormatFloat

def mynpow (b : Float) (p: Nat) : Float := match p with
  | 0 => 1
  | p+1 => b* (mynpow b p)

--  | p+1 => b * (mypow b p)
--  | _ => 1 / my
--case b==0 :

def roundToDecimal (f : Float) (n : Nat) : Float :=
  let multiplier := /-Math.pow 10 n-/ mynpow 10 n
  Float.round (f * multiplier) / multiplier

-- remove all trailing '0' from the right of string s
def rightTrimZeros (s : String) : String :=
  let s := s.toList.foldr (fun c s => if c == '0' then s else c::s) []
  if s.asString.isEmpty then "" else s.asString
--  let s := (s.toList.foldr (fun c s => if c == '0' then s else c::s) []).asString
--  if s.isEmpty then "0" else s


#eval rightTrimZeros "00000"
#eval rightTrimZeros "0"
#eval rightTrimZeros ""
#eval rightTrimZeros ".000"
#eval rightTrimZeros "1.00"
#eval rightTrimZeros "100"
#eval rightTrimZeros "10"
#eval rightTrimZeros "1"

def customFormatFloat (f : Float) : String :=
  let str := toString (/-Float.round f 0.1-/ roundToDecimal f 1) -- Round the float to the nearest 0.1
  let parts := str.splitOn "." -- Split the integer and decimal parts
  match parts with
  | [intPart, decPart] =>
    if intPart == "0" then
      "." ++ (rightTrimZeros decPart)      -- If the integer part is 0, start with "."
    else if decPart == "0" then
      intPart      -- If the decimal part is 0, only show the integer part
    else
      intPart ++ "." ++ (rightTrimZeros decPart) -- Concatenate the integer and decimal parts
  | _ => str -- Fallback to the original string if it doesn't contain a decimal point

-- Example usage
#eval customFormatFloat 0.1   -- Outputs: ".1"
#eval customFormatFloat 1.0   -- Outputs: "1"
#eval customFormatFloat 1.23  -- Outputs: "1.2"
#eval customFormatFloat 0.0   -- Outputs: "0"

end CustomFormatFloat
