# structure :

- bitboard mapping file
- movegeneration file
- evaluation file
- input / output file

# project pipeline :

   input_position -> convert_to_bitboards -> output_bitboards for each piece 
-> space of legal moves -> evaluate position -> return best move


# evaluation function prequisits:  

- While making and unmaking moves, it updates the material score for each  
  side, subdivided to pieces and pawns
- It updates the value derived from the piece-square tables for all pieces 
  except the king in the same manner
- When asked, it can return the position of either king
- When asked, it can return the number of white/black pawns, bishops, knights 
  or rooks
- It possesses a function int isPiece(int cl, int sq, int pc), returning 1 if 
  a piece pc of color cl stands on the square sq and 0 otherwise.

