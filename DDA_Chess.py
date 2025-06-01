import tkinter as tk
from PIL import Image, ImageTk
import chess
import chess.engine
import chess.svg
import cairosvg # type: ignore
import io
import joblib
import pandas as pd
import numpy as np
from collections import deque
import time
from sklearn.preprocessing import StandardScaler
import math
import pygame  # Add pygame for sound effect
import random

# Define global variables
global board, total_moves, total_accuracy, STOCKFISH_SKILL_LEVEL
global current_estimated_rating, current_win_probability
global last_move_type, prev_accuracy, prev_win_probability
global rating_history, win_probability_history, skill_level_history
global move_classifications, selected_square
global game_start_time, game_elapsed_time
global current_score  # NEW: Track the current evaluation score
# NEW: Set target win probability to 70%
TARGET_WIN_PROBABILITY = 0.7
# Load the trained models - UPDATED PATHS TO YOUR NEW MODELS
rating_model = joblib.load("./Models/100knewrating_model.pkl")
win_model = joblib.load("./Models/100knewWin_model.pkl")
scaler = StandardScaler()  # Also load the scaler that was saved with the models

board = chess.Board()

stockfish_path = './Stockfish/stockfish/stockfish-windows-x86-64-avx2.exe'
STOCKFISH_CONFIG = {'time': 0.0001, 'depth': 20}
STOCKFISH_SKILL_LEVEL = 10  # (0 to 20)

# Track player performance metrics
total_moves = 0
total_accuracy = 0
current_estimated_rating = 1500  # Default starting rating
current_win_probability = 0.5  # Default starting win probability
current_score = 0  # NEW: Initialize current score
move_classifications = {
    "Advantage": 0,
    "Slight Advantage": 0,
    "Neutral": 0,
    "Slight Disadvantage": 0,
    "Disadvantage": 0
}

# Time tracking variables
game_start_time = None
game_elapsed_time = None

# For mapping between old and new classification names (for compatibility with existing code)
classification_map = {
    "Brilliant": "Advantage",
    "Best": "Slight Advantage",
    "Good": "Neutral",
    "Inaccuracy": "Slight Disadvantage",
    "Blunder": "Disadvantage"
}
reverse_classification_map = {v: k for k, v in classification_map.items()}

# For tracking recent moves
last_move_type = None
prev_accuracy = 0
prev_win_probability = 0.5

# History for tracking performance trends
rating_history = []
win_probability_history = []
skill_level_history = []

selected_square = None


# Initialize pygame mixer for sound effects
pygame.mixer.init()

# Load sound effects
try:
    move_sound = pygame.mixer.Sound("./sounds/move.wav")  # Sound for player moves
    capture_sound = pygame.mixer.Sound("./sounds/capture.wav")  # Sound for captures
    check_sound = pygame.mixer.Sound("./sounds/check.wav")  # Sound for checks
    castle_sound = pygame.mixer.Sound("./sounds/castle.wav")  # Sound for castling
except:
    print("Warning: Could not load sound files. Make sure they exist in a 'sounds' folder.")
    # Create dummy sound objects that do nothing if files aren't found
    class DummySound:
        def play(self): pass
    move_sound = DummySound()
    capture_sound = DummySound()
    check_sound = DummySound()
    castle_sound = DummySound()

# Function to play the appropriate sound for a move
def play_move_sound(board, move):
    """Play the appropriate sound effect for a move."""
    # Check for castling
    if board.is_castling(move):
        castle_sound.play()
    # Check for captures
    elif board.is_capture(move):
        capture_sound.play()
    # Play regular move sound
    else:
        move_sound.play()
    
    # If the move results in check, play check sound
    board_copy = board.copy()
    board_copy.push(move)
    if board_copy.is_check():
        check_sound.play()


def classify_move(before_eval, after_eval):
    """Classify the player's move based on the change in evaluation scores."""
    if isinstance(before_eval, chess.engine.Cp) and isinstance(after_eval, chess.engine.Cp):
        # Calculate the change in evaluation
        # A positive change means the position improved for the player who just moved
        change = -after_eval.score() - before_eval.score()
        print(f"Score change: {change}")
        print(f"before_eval: {before_eval}")
        print(f"after_eval: {after_eval}")
        
        # Apply the classification thresholds and map to new names
        if change >= 200:  # Significant positive change
            return "Advantage"
        elif 70 < change < 200:  # Moderate positive change
            return "Slight Advantage"
        elif -70 <= change <= 70:  # Small positive or no change
            return "Neutral"
        elif -200 < change < -70:  # Minor negative change
            return "Slight Disadvantage"
        else:  # Significant negative change
            return "Disadvantage"
    
    # Handle mate scenarios
    elif isinstance(before_eval, chess.engine.Mate) and isinstance(after_eval, chess.engine.Mate):
        if before_eval.mate() < 0 and after_eval.mate() > 0:
            return "Advantage"  # Escaped from being mated to having mate
        elif before_eval.mate() > 0 and after_eval.mate() > 0:
            if abs(after_eval.mate()) < abs(before_eval.mate()):
                return "Slight Advantage"  # Found faster mate
            else:
                return "Neutral"  # Maintained mate
        elif before_eval.mate() < 0 and after_eval.mate() < 0:
            if abs(after_eval.mate()) > abs(before_eval.mate()):
                return "Slight Disadvantage"  # Getting mated faster
            else:
                return "Neutral"  # Delayed mate
        else:
            return "Neutral"
    
    # Handle transitions between mate and centipawn scores
    elif isinstance(before_eval, chess.engine.Cp) and isinstance(after_eval, chess.engine.Mate):
        if after_eval.mate() > 0:
            return "Advantage"  # Found a mate from non-mate position
        else:
            return "Disadvantage"  # Got mated from non-mate position
    elif isinstance(before_eval, chess.engine.Mate) and isinstance(after_eval, chess.engine.Cp):
        if before_eval.mate() < 0:
            return "Slight Advantage"  # Escaped from being mated
        else:
            return "Disadvantage"  # Lost a mate opportunity
    
    # Default for any other case
    return "Neutral"

def calculate_move_accuracy(before_eval, after_eval, move_number):
    """Calculate move accuracy based on the change in evaluation."""
    if isinstance(before_eval, chess.engine.Cp) and isinstance(after_eval, chess.engine.Cp):
        # Calculate the change in evaluation
        change = before_eval.score() - after_eval.score()
        
        # For opening moves, use a more sensitive scale
        if move_number <= 10:
            if change >= 0:  # Position improved or stayed the same
                return 100.0
            
            # Convert to absolute value for calculation
            error = abs(change)
            
            if error < 10:
                return 95.0
            elif error < 30:
                return 90.0 - (error - 10) / 20.0 * 10.0
            elif error < 50:
                return 80.0 - (error - 30) / 20.0 * 10.0
            else:
                # Fall through to regular calculation for larger errors
                pass
        
        # Regular accuracy calculation for middlegame/endgame
        if change >= 0:  # Position improved or stayed the same
            return 100.0
        
        # Convert to absolute value for calculation
        error = abs(change)
        
        # Scale accuracy based on the size of the error
        if error < 10:
            return 98.0
        elif error < 50:
            return 95.0 - (error - 10) / 40.0 * 10.0
        elif error < 100:
            return 85.0 - (error - 50) / 50.0 * 15.0
        elif error < 200:
            return 70.0 - (error - 100) / 100.0 * 20.0
        elif error < 500:
            return 50.0 - (error - 200) / 300.0 * 40.0
        else:
            return max(0, 10.0 - (error - 500) / 500.0 * 10.0)
    
    # Handle mate scenarios
    elif isinstance(before_eval, chess.engine.Mate) and isinstance(after_eval, chess.engine.Mate):
        if before_eval.mate() < 0 and after_eval.mate() > 0:
            return 100.0  # Escaped from being mated to having mate
        elif before_eval.mate() > 0 and after_eval.mate() > 0:
            if abs(after_eval.mate()) <= abs(before_eval.mate()):
                return 100.0  # Found same or faster mate
            else:
                diff = abs(after_eval.mate()) - abs(before_eval.mate())
                return max(90.0, 100.0 - diff * 2.0)  # Slower mate
        elif before_eval.mate() < 0 and after_eval.mate() < 0:
            if abs(after_eval.mate()) >= abs(before_eval.mate()):
                return 20.0  # Getting mated same or faster
            else:
                diff = abs(before_eval.mate()) - abs(after_eval.mate())
                return max(20.0, 40.0 - diff * 5.0)  # Delayed mate
        else:
            return 50.0
    
    # Handle transitions between mate and centipawn scores
    elif isinstance(before_eval, chess.engine.Cp) and isinstance(after_eval, chess.engine.Mate):
        if after_eval.mate() > 0:
            return 100.0  # Found a mate from non-mate position
        else:
            return 0.0  # Got mated from non-mate position
    elif isinstance(before_eval, chess.engine.Mate) and isinstance(after_eval, chess.engine.Cp):
        if before_eval.mate() < 0:
            return 90.0  # Escaped from being mated
        else:
            return 10.0  # Lost a mate opportunity
    
    # Default for other cases
    return 50.0



def enhanced_win_probability_estimation(features, current_score, avg_accuracy):
    """
    Enhanced win probability estimation that incorporates the current score and accuracy.
    
    Args:
        features (dict): Features of the current game state
        current_score (float): Current evaluation score (in centipawns)
        avg_accuracy (float): Average accuracy across all moves
        
    Returns:
        float: Enhanced win probability estimate
    """
    # First get the base win probability from the model
    feature_df = pd.DataFrame([features])
    
    # Calculate the ratios
    if feature_df['Moves'].iloc[0] > 0:  # Avoid division by zero
        feature_df['BrilliantRatio'] = feature_df['Brilliant'] / feature_df['Moves']
        feature_df['BestRatio'] = feature_df['Best'] / feature_df['Moves']
        feature_df['GoodRatio'] = feature_df['Good'] / feature_df['Moves']
        feature_df['InaccuracyRatio'] = feature_df['Inaccuracy'] / feature_df['Moves']
        feature_df['BlunderRatio'] = feature_df['Blunder'] / feature_df['Moves']
        
        # Calculate quality score
        feature_df['QualityScore'] = (feature_df['Brilliant'] * 2 + 
                                     feature_df['Best'] * 1 + 
                                     feature_df['Good'] * 0.5 - 
                                     feature_df['Inaccuracy'] * 1 - 
                                     feature_df['Blunder'] * 2) / feature_df['Moves']
    else:
        # Initialize with zeros for the first move
        feature_df['BrilliantRatio'] = 0
        feature_df['BestRatio'] = 0
        feature_df['GoodRatio'] = 0
        feature_df['InaccuracyRatio'] = 0
        feature_df['BlunderRatio'] = 0
        feature_df['QualityScore'] = 0
    
    # Get base win probability from model
    win_proba = win_model.predict_proba(feature_df)
    base_win_probability = win_proba[0][1] if win_proba.shape[1] > 1 else win_proba[0][0]
    
    # Initialize the score and accuracy adjustments
    score_adjustment = 0.0
    accuracy_adjustment = 0.0
    
    # NEW: Adjust win probability based on current score
    if isinstance(current_score, (int, float)):
        # Convert centipawns to win probability adjustment using a sigmoid function
        # This gives a smooth transition from 0 to 1
        # Scale factor determines how quickly probability changes with score
        scale_factor = 0.01
        score_adjustment = 1 / (1 + math.exp(-scale_factor * current_score)) - 0.5
    
    # NEW: Adjust win probability based on accuracy
    if avg_accuracy > 0:
        # High accuracy increases win probability, low accuracy decreases it
        if avg_accuracy > 85:
            accuracy_adjustment = 0.1
        elif avg_accuracy > 70:
            accuracy_adjustment = 0.05
        elif avg_accuracy < 50:
            accuracy_adjustment = -0.1
        elif avg_accuracy < 60:
            accuracy_adjustment = -0.05
    
    # Combine the base probability with the adjustments
    distance_from_target = TARGET_WIN_PROBABILITY - base_win_probability
    target_adjustment = distance_from_target * 0.3  # Scale factor controls strength of bias

    # Combine adjustments
    adjusted_win_probability = base_win_probability + score_adjustment + accuracy_adjustment + target_adjustment
    
    # Ensure the probability stays within [0, 1]
    adjusted_win_probability = max(0.0, min(1.0, adjusted_win_probability))
    
    print(f"Win probability: Base={base_win_probability:.2f}, Score Adj={score_adjustment:.2f}, Accuracy Adj={accuracy_adjustment:.2f}")
    print(f"Final adjusted win probability: {adjusted_win_probability:.2f}")
    
    return adjusted_win_probability

def enhanced_skill_adjustment(win_probability, estimated_rating, current_skill_level, move_classes, current_accuracy, avg_accuracy, current_score):
    """
    Enhanced Stockfish skill level adjustment based on multiple factors including accuracy and score.
    
    Args:
        win_probability (float): Probability of the player winning
        estimated_rating (float): Estimated player rating
        current_skill_level (int): Current Stockfish skill level
        move_classes (dict): Dictionary of move classifications
        current_accuracy (float): Accuracy of the latest move
        avg_accuracy (float): Average accuracy across all moves
        current_score (float): Current evaluation score from the score bar (in centipawns)
        
    Returns:
        int: Adjusted Stockfish skill level
    """
    # Define rating-based target win probability (around 50% for balanced play)
    target_win_prob = TARGET_WIN_PROBABILITY
    
    # Map new classification names to original ones for compatibility
    good_moves = move_classes['Advantage'] + move_classes['Slight Advantage'] + move_classes['Neutral']
    bad_moves = move_classes['Slight Disadvantage'] + move_classes['Disadvantage']
    move_ratio = 1.0 if good_moves + bad_moves == 0 else good_moves / max(1, good_moves + bad_moves)
    
    # Calculate skill adjustments based on various factors
    # Adjust based on difference from TARGET win probability
    win_prob_adjustment = 3 * (target_win_prob - win_probability)  # Stronger adjustment    
    
    # Analyze trend in win probability if we have enough history
    trend_adjustment = 0
    if len(win_probability_history) >= 3:
        recent_trend = np.mean(np.diff(win_probability_history[-3:]))
        trend_adjustment = -2 * recent_trend  # Counteract rapid changes in win probability
    
    # Move quality adjustment
    move_quality_adjustment = (0.5 - move_ratio) * 4  # Adjust harder if making more bad moves
    
    # NEW: Accuracy-based adjustment
    accuracy_adjustment = 0
    if avg_accuracy > 0:
        # If player has high accuracy, increase difficulty
        if avg_accuracy > 85:
            accuracy_adjustment = 2
        elif avg_accuracy > 70:
            accuracy_adjustment = 1
        # If player has low accuracy, decrease difficulty
        elif avg_accuracy < 60:
            accuracy_adjustment = -1
        elif avg_accuracy < 45:
            accuracy_adjustment = -2
    
    # NEW: Score-based adjustment - if player is losing badly, reduce difficulty
    score_adjustment = 0
    if isinstance(current_score, (int, float)):
        # Score is in centipawns from the player's perspective
        # Negative means the player is losing, positive means winning
        if current_score < -300:  # Player losing badly
            score_adjustment = -2
        elif current_score < -150:  # Player losing moderately
            score_adjustment = -1
        elif current_score > 300:  # Player winning easily
            score_adjustment = 2
        elif current_score > 150:  # Player winning moderately
            score_adjustment = 1
    
    # Combine all adjustments
    total_adjustment = (
        win_prob_adjustment + 
        trend_adjustment + 
        move_quality_adjustment + 
        accuracy_adjustment +  # NEW: Include accuracy adjustment
        score_adjustment       # NEW: Include score adjustment
    )
    
    # Scale the adjustment based on current skill level (larger adjustments in mid-range)
    scaling_factor = 1.0
    if current_skill_level < 5 or current_skill_level > 15:
        scaling_factor = 0.5  # Smaller adjustments at extreme skill levels
    
    # Calculate the final adjustment, rounded to nearest integer
    skill_change = round(total_adjustment * scaling_factor)
    
    # Apply the change with bounds checking
    new_skill_level = max(1, min(20, current_skill_level + skill_change))
    
    # Prevent constant flip-flopping by requiring stronger evidence for reversing direction
    if len(skill_level_history) >= 2:
        last_change = skill_level_history[-1] - skill_level_history[-2]
        if last_change * skill_change < 0 and abs(skill_change) <= 1:  # If reversing direction with small change
            new_skill_level = current_skill_level  # Maintain current level to avoid oscillation
    
    # Print debugging information
    print(f"Win Prob: {win_probability:.2f}, Rating: {estimated_rating:.0f}, Accuracy: {avg_accuracy:.2f}%")
    print(f"Adjustments - Win: {win_prob_adjustment:.2f}, Trend: {trend_adjustment:.2f}, Quality: {move_quality_adjustment:.2f}")
    print(f"New Adjustments - Accuracy: {accuracy_adjustment:.2f}, Score: {score_adjustment:.2f}")
    print(f"Total adjustment: {skill_change} (scaled by {scaling_factor})")
    
    return new_skill_level

# Modified evaluate function to play sound for engine moves
def evaluate(board, return_mate_n=False):
    """Use Stockfish to make a move and update the display."""
    global STOCKFISH_SKILL_LEVEL, current_score
    
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        engine.configure({"Skill Level": STOCKFISH_SKILL_LEVEL})
        
        

        # NEW: Occasionally have the engine make a deliberate mistake
        should_make_mistake = random.random() <  random.uniform(1 - current_win_probability,0)   # 20% chance of making a mistake
        
        # Analyze current position
        info = engine.analyse(board, chess.engine.Limit(**STOCKFISH_CONFIG))
        score = info['score'].relative
        
        # Store the current score for later use in win probability estimation
        if isinstance(score, chess.engine.Cp):
            current_score = score.score()
        else:
            # Handle mate scores - convert to large centipawn values
            if score.mate() > 0:
                current_score = 10000 - (score.mate() * 100)  # Positive large value
            else:
                current_score = -10000 - (score.mate() * 100)  # Negative large value
        
        print(f"Current position score: {score} (stored as {current_score})")
        
        # Check game state before making engine move
        if board.is_game_over():
            display_game_result(board)
            return str(score)
        
        # Let engine make its move
        if should_make_mistake and len(list(board.legal_moves)) > 3:
            # Force engine to make a suboptimal move
            legal_moves = list(board.legal_moves)
            
            # Analyze top moves
            move_scores = []
            for move in legal_moves[:min(5, len(legal_moves))]:
                board_copy = board.copy()
                board_copy.push(move)
                result = engine.analyse(board_copy, chess.engine.Limit(depth=10, time=0.001))
                # Negate score because we're evaluating from opponent's perspective
                move_score = -result['score'].relative.score() if isinstance(result['score'].relative, chess.engine.Cp) else 0
                move_scores.append((move, move_score))
            
            # Sort moves by score (best to worst)
            move_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Choose a suboptimal move (not the worst, but not the best)
            if len(move_scores) >= 3:
                # Choose randomly from the middle range of moves
                start_idx = len(move_scores) // 3
                end_idx = 2 * len(move_scores) // 3
                engine_move = move_scores[random.randint(start_idx, end_idx)][0]
            else:
                # If few options, just pick randomly
                engine_move = random.choice(legal_moves)
                
            print(f"Engine deliberately making suboptimal move: {engine_move}")
        else:
            result = engine.play(board, chess.engine.Limit(**STOCKFISH_CONFIG))
            engine_move = result.move
            print(f"Engine playing normal move: {engine_move}")
        
        # Play sound for the engine's move BEFORE pushing it
        play_move_sound(board, engine_move)
        
        # Push the engine's move
        board.push(engine_move)
        
        # Analyze position after engine's move
        info = engine.analyse(board, chess.engine.Limit(**STOCKFISH_CONFIG))
        score = info['score'].relative
        
        # Update the board display
        draw_board(canvas, board, last_move=engine_move)
        
        # Update the score bar
        update_score_bar(score)
        
        # Check if the game is over after engine's move
        if board.is_game_over():
            display_game_result(board)
        
        print(f"Position after engine move: ")
        print(f"{board}")
    
    return str(score)

# Function to make the engine move after a delay
def delayed_engine_move():
    """Makes the engine move after a delay."""
    engine_move = evaluate(board)
    # The sound for the engine move will be played inside the evaluate function

# def draw_board(canvas, board, valid_moves=None, last_move=None):
#     if valid_moves is None:
#         valid_moves = []
#     squares = chess.SquareSet(valid_moves)
#     if last_move:
#         squares.add(last_move.from_square)
#         squares.add(last_move.to_square)
#     svg_board = chess.svg.board(board=board, squares=squares, size=400)
#     png_data = cairosvg.svg2png(bytestring=svg_board.encode('utf-8'))
#     image = Image.open(io.BytesIO(png_data))
#     photo = ImageTk.PhotoImage(image)
#     canvas.image = photo  # Keep a reference to avoid garbage collection
#     canvas.create_image(0, 0, anchor=tk.NW, image=photo)

def draw_board(canvas, board, valid_moves=None, last_move=None):
    if valid_moves is None:
        valid_moves = []
    
    # Create square sets for different highlights
    valid_squares = chess.SquareSet(valid_moves)
    last_move_squares = chess.SquareSet()
    
    # Add last move squares if available
    if last_move:
        last_move_squares.add(last_move.from_square)
        last_move_squares.add(last_move.to_square)
    
    # Create color dictionary for custom highlighting
    colors = {}
    
    # Color for valid moves (light green)
    for square in valid_squares:
        colors[square] = "#90EE90"  # Light green
    
    # Color for last move squares (orange/amber - overwrites valid moves if there's overlap)
    for square in last_move_squares:
        colors[square] = "#FFB347"  # Orange/amber
    
    # Create the SVG with custom colors
    svg_board = chess.svg.board(
        board=board, 
        size=400,
        fill=colors,  # Apply our custom color dictionary
        coordinates=True,  # Show board coordinates (a-h, 1-8)
        check=board.king(board.turn) if board.is_check() else None  # Highlight check
    )
    
    # Convert SVG to PNG for display
    png_data = cairosvg.svg2png(bytestring=svg_board.encode('utf-8'))
    image = Image.open(io.BytesIO(png_data))
    photo = ImageTk.PhotoImage(image)
    canvas.image = photo  # Keep a reference to avoid garbage collection
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

def on_square_click(event):
    global selected_square, total_moves, total_accuracy, STOCKFISH_SKILL_LEVEL
    global current_estimated_rating, current_win_probability
    global last_move_type, prev_accuracy, prev_win_probability
    global game_start_time
    
    # Initialize game start time if this is the first move
    if game_start_time is None:
        game_start_time = time.time()
        update_time_elapsed()
    
    # Don't process clicks if game is over
    if board.is_game_over():
        return
    
    col = event.x // 50
    row = 7 - (event.y // 50)
    square = chess.square(col, row)
    piece = board.piece_at(square)
    
    if selected_square is None:
        if piece is not None and piece.color == board.turn:  # Only select pieces of current player's turn
            selected_square = square
            valid_moves = [move.to_square for move in board.legal_moves if move.from_square == selected_square]
            draw_board(canvas, board, valid_moves)
    else:
        # Check if it's a legal move
        move = chess.Move(from_square=selected_square, to_square=square)
        promotion_move = False
        
        # Check if it's a promotion move
        moving_piece = board.piece_at(selected_square)
        if (moving_piece and moving_piece.piece_type == chess.PAWN and 
            ((board.turn == chess.WHITE and chess.square_rank(square) == 7) or
             (board.turn == chess.BLACK and chess.square_rank(square) == 0))):
            promotion_move = True
        
        # Check if the base move (without promotion) is legal
        base_move_legal = move in board.legal_moves or any(
            m.from_square == selected_square and m.to_square == square and m.promotion
            for m in board.legal_moves
        )
        
        if base_move_legal:
            if promotion_move:
                # Create a promotion dialog
                show_promotion_dialog(selected_square, square)
                selected_square = None
                return
            else:
                # Process non-promotion move
                process_move(move)
        else:
            # Handle selecting a new piece or deselecting
            if piece is not None and piece.color == board.turn:
                selected_square = square
                valid_moves = [move.to_square for move in board.legal_moves if move.from_square == selected_square]
                draw_board(canvas, board, valid_moves)
            else:
                selected_square = None
                draw_board(canvas, board)

def show_promotion_dialog(from_square, to_square):
    """Display a dialog to choose promotion piece."""
    promotion_window = tk.Toplevel(root)
    promotion_window.title("Choose Promotion")
    promotion_window.geometry("300x100")
    promotion_window.configure(bg="#f0f0f0")
    promotion_window.transient(root)
    promotion_window.grab_set()
    
    # Prevent closing the window with the X button
    promotion_window.protocol("WM_DELETE_WINDOW", lambda: None)
    
    prompt = tk.Label(promotion_window, text="Choose a piece for promotion:", 
                     font=("Helvetica", 12), bg="#f0f0f0")
    prompt.pack(pady=5)
    
    buttons_frame = tk.Frame(promotion_window, bg="#f0f0f0")
    buttons_frame.pack(pady=5)
    
    # Button for each promotion option
    queen_button = tk.Button(buttons_frame, text="Queen", width=8, 
                            command=lambda: select_promotion(promotion_window, from_square, to_square, chess.QUEEN))
    queen_button.pack(side=tk.LEFT, padx=5)
    
    rook_button = tk.Button(buttons_frame, text="Rook", width=8,
                           command=lambda: select_promotion(promotion_window, from_square, to_square, chess.ROOK))
    rook_button.pack(side=tk.LEFT, padx=5)
    
    bishop_button = tk.Button(buttons_frame, text="Bishop", width=8,
                             command=lambda: select_promotion(promotion_window, from_square, to_square, chess.BISHOP))
    bishop_button.pack(side=tk.LEFT, padx=5)
    
    knight_button = tk.Button(buttons_frame, text="Knight", width=8,
                             command=lambda: select_promotion(promotion_window, from_square, to_square, chess.KNIGHT))
    knight_button.pack(side=tk.LEFT, padx=5)

def select_promotion(window, from_square, to_square, piece_type):
    """Handle the promotion piece selection."""
    move = chess.Move(from_square=from_square, to_square=to_square, promotion=piece_type)
    window.destroy()
    process_move(move)


# Modified process_move function to play sounds
def process_move(move):
    """Process a chess move after it's been validated and promotion is handled."""
    global total_moves, total_accuracy, STOCKFISH_SKILL_LEVEL
    global current_estimated_rating, current_win_probability
    global last_move_type, prev_accuracy, prev_win_probability
    global current_score
    
    # Play sound for player's move
    play_move_sound(board, move)
    
    # Evaluate player's move using before/after evaluation
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        # Use maximum skill level for evaluation
        engine.configure({"Skill Level": 20})
        
        print("\n----- Move Analysis -----")
        print(f"Current position FEN: {board.fen()}")
        print(f"Player is moving: {move}")
        print(f"Move number: {total_moves + 1}")
        
        # Get evaluation BEFORE the move
        before_info = engine.analyse(board, chess.engine.Limit(**STOCKFISH_CONFIG))
        before_eval = before_info['score'].relative
        print(f"Evaluation BEFORE move: {before_eval}")
        
        # Make the player's move
        board.push(move)
        
        # Get evaluation AFTER the move (from opponent's perspective)
        after_info = engine.analyse(board, chess.engine.Limit(**STOCKFISH_CONFIG))
        # Need to negate this score since it's from opponent's perspective
        after_eval = after_info['score'].relative
        print(f"Evaluation AFTER move (opponent perspective): {after_eval}")
        
        # Undo the move for now (we'll push it again later)
        board.pop()
        
        # Classify the move using the before/after evals
        move_type = classify_move(before_eval, after_eval)
        # Save this move type for indicators
        last_move_type = move_type
        
        move_classifications[move_type] += 1
        
        # Calculate accuracy
        current_accuracy = calculate_move_accuracy(before_eval, after_eval, total_moves + 1)
        print(f"Move classification: {move_type}, Accuracy: {current_accuracy:.2f}%")
        
        # Get the current score for win probability and difficulty adjustment
        # We'll use the evaluation before the move
        if isinstance(before_eval, chess.engine.Cp):
            current_score = before_eval.score()
        else:
            # Handle mate scores - convert to large centipawn values
            if before_eval.mate() > 0:
                current_score = 10000 - (before_eval.mate() * 100)  # Positive large value
            else:
                current_score = -10000 - (before_eval.mate() * 100)  # Negative large value
        
        # Update previous values for trend indicators
        prev_win_probability = current_win_probability
        prev_accuracy = current_accuracy if total_moves > 0 else current_accuracy
        
        total_moves += 1
        total_accuracy += current_accuracy
        avg_accuracy = total_accuracy / total_moves if total_moves > 0 else 0
        
        # Extract features for the current game state
        # Convert new classification names to original names for model compatibility
        features = {
            "Moves": total_moves,
            "Brilliant": move_classifications["Advantage"],
            "Best": move_classifications["Slight Advantage"],
            "Good": move_classifications["Neutral"],
            "Inaccuracy": move_classifications["Slight Disadvantage"],
            "Blunder": move_classifications["Disadvantage"],
            "BrilliantRatio": move_classifications["Advantage"] / max(1, total_moves),
            "BestRatio": move_classifications["Slight Advantage"] / max(1, total_moves),
            "GoodRatio": move_classifications["Neutral"] / max(1, total_moves),
            "InaccuracyRatio": move_classifications["Slight Disadvantage"] / max(1, total_moves),
            "BlunderRatio": move_classifications["Disadvantage"] / max(1, total_moves),
            "QualityScore": (move_classifications["Advantage"] * 2 + 
                           move_classifications["Slight Advantage"] * 1 + 
                           move_classifications["Neutral"] * 0.5 - 
                           move_classifications["Slight Disadvantage"] * 1 - 
                           move_classifications["Disadvantage"] * 2) / max(1, total_moves)
        }
        
        # Ensure we have a DataFrame with all required features
        features_df = pd.DataFrame([features])
        
        # Estimate player rating using the original model
        try:
            estimated_rating = rating_model.predict(features_df)[0]
            current_estimated_rating = estimated_rating
            
            # Use enhanced win probability estimation
            current_win_probability = enhanced_win_probability_estimation(features, current_score, avg_accuracy)
            
            # Update history for trend analysis
            rating_history.append(estimated_rating)
            win_probability_history.append(current_win_probability)
            
            # Update the display
            update_move_classifications()
            update_accuracy_display(current_accuracy)
            # update_rating_display(estimated_rating)
            update_win_probability_display(current_win_probability)
            
            # Adjust Stockfish skill level with enhanced function
            STOCKFISH_SKILL_LEVEL = enhanced_skill_adjustment(
                current_win_probability, 
                estimated_rating, 
                STOCKFISH_SKILL_LEVEL,
                move_classifications,
                current_accuracy,
                avg_accuracy,
                current_score
            )
        except ValueError as e:
            # Handle model error more gracefully
            print(f"Error in prediction: {e}")
            print("Continuing with default values...")
            # Use default values if model fails
            estimated_rating = current_estimated_rating
            current_win_probability = 0.5
            
        skill_level_history.append(STOCKFISH_SKILL_LEVEL)
        update_skill_level_display(STOCKFISH_SKILL_LEVEL)
        engine.configure({"Skill Level": 20})
        info = engine.analyse(board, chess.engine.Limit(**STOCKFISH_CONFIG))
        score = info['score'].relative
        engine.configure({"Skill Level": STOCKFISH_SKILL_LEVEL})

    # Push the player's move and update the board
    board.push(move)
    draw_board(canvas, board, last_move=move)
    update_score_bar(score)
    # Check if the game is over
    if board.is_game_over():
        display_game_result(board)
    else:
        # Let the engine respond if game is not over, but with a delay
        # Schedule the engine move with a random delay between 1-3 seconds
        root.after(random.randint(1000, 3000), lambda: delayed_engine_move())


def update_score_bar(score):
    """Update the score evaluation bar with proper scaling and display."""
    score_canvas.delete("all")
    bar_height = 200  # Match the canvas height
    bar_width = 50    # Match the canvas width
    center = bar_height // 2
    
    # Determine actual score value with bounds
    if score.is_mate():
        if score.mate() > 0:  # Checkmate for the current side
            score_value = 1000  # Max advantage
        else:  # Checkmate against the current side
            score_value = -1000  # Max disadvantage
    else:
        # Clamp score to reasonable range to avoid visual oddities
        raw_score = score.score()
        score_value = max(-1000, min(1000, raw_score))
    
    # Map score from [-1000, 1000] to [0, bar_height] for visual display
    # 0 centipawns (equal position) should be at the center line
    normalized_value = (score_value + 1000) / 2000  # Maps to [0,1]
    bar_position = int((1 - normalized_value) * bar_height)  # Inverted for display
    
    # Draw the bar (black on top for negative scores, white on bottom for positive)
    score_canvas.create_rectangle(0, 0, bar_width, bar_position, fill="black", outline="")
    score_canvas.create_rectangle(0, bar_position, bar_width, bar_height, fill="white", outline="")
    
    # Add a center line to mark equal position
    score_canvas.create_line(0, center, bar_width, center, fill="gray")
    
    # Update the score text (display in pawns, not centipawns)
    if score.is_mate():
        if score.mate() > 0:
            score_text = f"Mate in {score.mate()}"
        else:
            score_text = f"Mated in {abs(score.mate())}"
        score_label.config(text=score_text)
    else:
        score_label.config(text=f"Score: {score_value / 100:.2f}")  # Convert centipawns to pawns

def update_move_classifications():
    """Update the move classification labels and show the last move indicator."""
    # First update the counts for each classification
    advantage_label.config(text=f"Advantage: {move_classifications['Advantage']}")
    slight_advantage_label.config(text=f"Slight Advantage: {move_classifications['Slight Advantage']}")
    neutral_label.config(text=f"Neutral: {move_classifications['Neutral']}")
    slight_disadvantage_label.config(text=f"Slight Disadvantage: {move_classifications['Slight Disadvantage']}")
    disadvantage_label.config(text=f"Disadvantage: {move_classifications['Disadvantage']}")
    
    # Clear all indicators first
    advantage_indicator.config(text="")
    slight_advantage_indicator.config(text="")
    neutral_indicator.config(text="")
    slight_disadvantage_indicator.config(text="")
    disadvantage_indicator.config(text="")
    
    # Show indicator for the last move type (upward arrow)
    if last_move_type == "Advantage":
        advantage_indicator.config(text="↑", fg="#007700")
    elif last_move_type == "Slight Advantage":
        slight_advantage_indicator.config(text="↑", fg="#0000FF")
    elif last_move_type == "Neutral":
        neutral_indicator.config(text="↑", fg="#00AAFF")
    elif last_move_type == "Slight Disadvantage":
        slight_disadvantage_indicator.config(text="↑", fg="#FF8800")
    elif last_move_type == "Disadvantage":
        disadvantage_indicator.config(text="↑", fg="#FF0000")

def update_accuracy_display(current_accuracy):
    """Update the accuracy display with trend indicator."""
    # Calculate the average accuracy
    avg_accuracy = total_accuracy / total_moves if total_moves > 0 else 0
    
    # Update the accuracy label
    accuracy_label.config(text=f"Accuracy: {avg_accuracy:.2f}%")
    
    # Add trend indicator based on current vs previous accuracy
    if total_moves > 1:
        if current_accuracy > prev_accuracy:
            # Accuracy improved
            accuracy_trend_indicator.config(text="↑", fg="green")
        elif current_accuracy < prev_accuracy:
            # Accuracy worsened
            accuracy_trend_indicator.config(text="↓", fg="red")
        else:
            # Accuracy unchanged
            accuracy_trend_indicator.config(text="→", fg="gray")
    else:
        # First move
        accuracy_trend_indicator.config(text="")

# def update_rating_display(rating):
#     """Update the estimated rating display."""
#     rating_label.config(text=f"Est. Rating: {rating:.0f}")
    
#     # Update rating trend indicator
#     if len(rating_history) > 1:
#         if rating > rating_history[-2]:
#             rating_trend_indicator.config(text="↑", fg="green")
#         elif rating < rating_history[-2]:
#             rating_trend_indicator.config(text="↓", fg="red")
#         else:
#             rating_trend_indicator.config(text="→", fg="gray")

def update_win_probability_display(win_prob):
    """Update the win probability display with trend indicator."""
    win_prob_label.config(text=f"Win Prob: {win_prob*100:.1f}%")
    
    # Update the win probability bar
    win_prob_canvas.delete("all")
    width = int(win_prob * 150)  # Scale to canvas width
    win_prob_canvas.create_rectangle(0, 0, width, 20, fill="green", outline="")
    win_prob_canvas.create_rectangle(width, 0, 150, 20, fill="white", outline="")
    
    # Add trend indicator
    if len(win_probability_history) > 1:
        diff = win_prob - prev_win_probability
        
        if abs(diff) > 0.01:  # More than 1% change
            if diff > 0:
                # Win probability improved
                win_prob_trend_indicator.config(text="↑", fg="green")
            else:
                # Win probability decreased
                win_prob_trend_indicator.config(text="↓", fg="red")
        else:
            # No significant change
            win_prob_trend_indicator.config(text="→", fg="gray")

def update_skill_level_display(skill_level):
    """Update the Stockfish skill level display."""
    # skill_level_label.config(text=f"Bot Level: {skill_level}")

def display_game_result(board):
    """Display the game result when the game is over."""
    global game_elapsed_time
    
    # Calculate elapsed time
    if game_start_time is not None:
        game_elapsed_time = time.time() - game_start_time
    
    # Stop the time update
    root.after_cancel(time_update_id)
    
    # Create a result window
    result_window = tk.Toplevel(root)
    result_window.title("Game Result")
    result_window.geometry("400x200")
    result_window.configure(bg="#f0f0f0")
    
    # Determine the result text and color
    result_text = ""
    text_color = "black"
    
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        result_text = f"{winner} Wins by Checkmate!"
        text_color = "#007700"  # Green
    elif board.is_stalemate():
        result_text = "Draw by Stalemate"
        text_color = "#0000FF"  # Blue
    elif board.is_insufficient_material():
        result_text = "Draw by Insufficient Material"
        text_color = "#0000FF"  # Blue
    elif board.is_seventyfive_moves():
        result_text = "Draw by 75-Move Rule"
        text_color = "#0000FF"  # Blue
    elif board.is_fivefold_repetition():
        result_text = "Draw by Fivefold Repetition"
        text_color = "#0000FF"  # Blue
    elif board.is_fifty_moves():
        result_text = "Draw by 50-Move Rule"
        text_color = "#0000FF"  # Blue
    elif board.is_repetition():
        result_text = "Draw by Repetition"
        text_color = "#0000FF"  # Blue
    else:
        result_text = "Game Over"
    
    # Format elapsed time
    if game_elapsed_time is not None:
        hours = int(game_elapsed_time // 3600)
        minutes = int((game_elapsed_time % 3600) // 60)
        seconds = int(game_elapsed_time % 60)
        time_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"
        time_text = f"Game Duration: {time_str}"
    else:
        time_text = "Game Duration: Unknown"
    
    # Create and place result label
    result_label = tk.Label(result_window, text=result_text, font=("Helvetica", 18, "bold"), 
                           fg=text_color, bg="#f0f0f0")
    result_label.pack(pady=20)
    
    # Create and place time label
    time_label = tk.Label(result_window, text=time_text, font=("Helvetica", 14), 
                           bg="#f0f0f0")
    time_label.pack(pady=10)
    
    # Create and place performance statistics
    stats_text = f"Your Moves: {total_moves//2}\nAverage Accuracy: {total_accuracy/total_moves:.2f}%\nEstimated Rating: {current_estimated_rating:.0f}"
    stats_label = tk.Label(result_window, text=stats_text, font=("Helvetica", 12), 
                          bg="#f0f0f0", justify=tk.LEFT)
    stats_label.pack(pady=10)
    
    # Create close button
    close_button = tk.Button(result_window, text="Close", font=("Helvetica", 10),
                            command=result_window.destroy)
    close_button.pack(pady=10)
    
    # Update the game result label in the main window
    game_result_label.config(text=result_text, fg=text_color)

def update_time_elapsed():
    """Update the elapsed time display."""
    global time_update_id
    
    if game_start_time is not None:
        elapsed = time.time() - game_start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        if hours > 0:
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            time_str = f"{minutes:02d}:{seconds:02d}"
        
        time_elapsed_label.config(text=f"Time: {time_str}")
    
    # Schedule the next update
    time_update_id = root.after(1000, update_time_elapsed)

def restart_game():
    """Reset the game to the starting position."""
    global board, total_moves, total_accuracy, move_classifications
    global current_estimated_rating, current_win_probability
    global last_move_type, prev_accuracy, prev_win_probability
    global game_start_time, game_elapsed_time, time_update_id
    global rating_history, win_probability_history, skill_level_history
    global current_score  # NEW: Reset current score
    
    # Reset the board
    board = chess.Board()
    
    # Reset performance metrics
    total_moves = 0
    total_accuracy = 0
    current_estimated_rating = 1500
    current_win_probability = 0.5
    last_move_type = None
    prev_accuracy = 0
    prev_win_probability = 0.5
    current_score = 0  # NEW: Reset the score
    
    # Reset history arrays
    rating_history = []
    win_probability_history = []
    skill_level_history = []
    
    # Reset time tracking
    game_start_time = None
    game_elapsed_time = None
    if 'time_update_id' in globals():
        root.after_cancel(time_update_id)
    time_elapsed_label.config(text="Time: 00:00")
    time_update_id = root.after(1000, update_time_elapsed)
    
    # Reset move classifications
    for key in move_classifications:
        move_classifications[key] = 0
    
    # Update displays
    update_move_classifications()
    accuracy_label.config(text=f"Accuracy: 0.00%")
    accuracy_trend_indicator.config(text="")
    # update_rating_display(current_estimated_rating)
    update_win_probability_display(current_win_probability)
    win_prob_trend_indicator.config(text="")
    game_result_label.config(text="")
    
    # Draw the new board
    draw_board(canvas, board)
    
    # Reset the score bar
    score_canvas.delete("all")
    center = 100  # Half of bar_height (200)
    score_canvas.create_rectangle(0, 0, 50, center, fill="black", outline="")
    score_canvas.create_rectangle(0, center, 50, 200, fill="white", outline="")
    score_canvas.create_line(0, center, 50, center, fill="gray")
    score_label.config(text="Score: 0.00")

def show_help():
    """Display help information for the chess application."""
    help_window = tk.Toplevel(root)
    help_window.title("Chess Application Help")
    help_window.geometry("500x500")  # Increased height to accommodate new feature descriptions
    help_window.configure(bg="#f0f0f0")
    
    help_text = """
    Chess Application Help
    
    Game Controls:
    - Click on a piece to select it
    - Click on a valid square to move the selected piece
    
    Score Bar:
    - The vertical bar shows the current evaluation
    - Black (top) means advantage for black
    - White (bottom) means advantage for white
    - Gray line in the middle represents equal position
    
    Move Classifications:
    - Advantage (Green): Exceptionally strong move that significantly improves position
    - Slight Advantage (Blue): Strong move that moderately improves position
    - Neutral (Light Blue): Solid move that maintains or slightly changes position
    - Slight Disadvantage (Orange): Weak move that slightly worsens position
    - Disadvantage (Red): Poor move that significantly worsens position
    - The upward arrow (↑) indicates your most recent move type
    
    Performance Indicators:
    - Arrows next to metrics show trends (↑ improving, ↓ worsening, → steady)
    - Accuracy shows how close your moves are to the best engine moves
    - Rating is an estimate of your current playing strength
    - Win Probability shows your chances of winning (based on move quality, accuracy and position)
    - Time shows the elapsed time since the game started
    
    Adaptive Difficulty:
    - The computer's playing strength adjusts based on your performance
    - Your move accuracy and the current position evaluation affect difficulty
    - The system tries to maintain a balanced challenge level
    
    Win Probability:
    - Calculated from move quality, current position score, and accuracy
    - Higher accuracy and better positions increase win probability
    - The progress bar shows your estimated chances graphically
    
    Game Results:
    - When the game ends, the result will be displayed
    - You'll see your final statistics and game duration
    
    To restart the game, click the "Restart" button.
    """
    
    help_label = tk.Label(help_window, text=help_text, font=("Helvetica", 11), 
                          bg="#f0f0f0", justify=tk.LEFT, padx=20, pady=20)
    help_label.pack(expand=True, fill=tk.BOTH)
    
    close_button = tk.Button(help_window, text="Close", font=("Helvetica", 10),
                             command=help_window.destroy)
    close_button.pack(side=tk.BOTTOM, pady=10)

# Create the main window
root = tk.Tk()
root.title("Adaptive Chess")
root.configure(bg="#f0f0f0")

# Create a frame for overall layout
main_frame = tk.Frame(root, bg="#f0f0f0", padx=10, pady=10)
main_frame.pack(expand=True, fill=tk.BOTH)

# Create a frame to hold the chessboard and score bar
game_frame = tk.Frame(main_frame, bg="#f0f0f0")
game_frame.pack(side=tk.LEFT, padx=5)

# Create a canvas to draw the chess board
canvas = tk.Canvas(game_frame, width=400, height=400)
canvas.pack(side=tk.TOP, pady=5)

# Create a frame for game metrics
metrics_frame = tk.Frame(main_frame, bg="#f0f0f0", padx=10)
metrics_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Score and analysis section
score_frame = tk.Frame(metrics_frame, bg="#f0f0f0", relief=tk.GROOVE, bd=2)
score_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

# Create a label to display the score
score_label = tk.Label(score_frame, text="Score: 0.00", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
score_label.pack(side=tk.TOP, anchor=tk.W, pady=2)

# Create a canvas to draw the score bar
score_canvas = tk.Canvas(score_frame, width=50, height=200, bg="white", highlightthickness=1, highlightbackground="#999")
score_canvas.pack(side=tk.LEFT, padx=5)

# Add a center line to the score bar (equal position)
score_canvas.create_line(0, 100, 50, 100, fill="gray")

# Game result and time display
status_frame = tk.Frame(metrics_frame, bg="#f0f0f0", relief=tk.GROOVE, bd=2)
status_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

# Game result display
game_result_label = tk.Label(status_frame, text="", font=("Helvetica", 12, "bold"), bg="#f0f0f0")
game_result_label.pack(side=tk.TOP, anchor=tk.W, pady=2)

# Time elapsed display
time_elapsed_label = tk.Label(status_frame, text="Time: 00:00", font=("Helvetica", 12), bg="#f0f0f0")
time_elapsed_label.pack(side=tk.TOP, anchor=tk.W, pady=2)

# Performance metrics section
performance_frame = tk.Frame(metrics_frame, bg="#f0f0f0", relief=tk.GROOVE, bd=2)
performance_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

# Player statistics
stats_frame = tk.Frame(performance_frame, bg="#f0f0f0")
stats_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

# Create a frame for accuracy display
accuracy_frame = tk.Frame(stats_frame, bg="#f0f0f0")
accuracy_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

# Accuracy label and trend indicator
accuracy_label = tk.Label(accuracy_frame, text="Accuracy: 0.00%", font=("Helvetica", 12), bg="#f0f0f0")
accuracy_label.pack(side=tk.LEFT)
accuracy_trend_indicator = tk.Label(accuracy_frame, text="", font=("Helvetica", 12, "bold"), bg="#f0f0f0")
accuracy_trend_indicator.pack(side=tk.LEFT, padx=5)

# Create a frame for rating and win probability
rating_frame = tk.Frame(stats_frame, bg="#f0f0f0")
rating_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

# # Rating display with trend indicator
# rating_label = tk.Label(rating_frame, text="Est. Rating: 1500", font=("Helvetica", 12), bg="#f0f0f0")
# rating_label.pack(side=tk.LEFT)
# rating_trend_indicator = tk.Label(rating_frame, text="", font=("Helvetica", 12, "bold"), bg="#f0f0f0")
# rating_trend_indicator.pack(side=tk.LEFT, padx=5)

# Win probability display
win_prob_frame = tk.Frame(stats_frame, bg="#f0f0f0")
win_prob_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

# Win probability label and trend indicator
win_prob_label = tk.Label(win_prob_frame, text="Win Prob: 50.0%", font=("Helvetica", 12), bg="#f0f0f0")
win_prob_label.pack(side=tk.LEFT)
win_prob_trend_indicator = tk.Label(win_prob_frame, text="", font=("Helvetica", 12, "bold"), bg="#f0f0f0")
win_prob_trend_indicator.pack(side=tk.LEFT, padx=5)

# Win probability bar
win_prob_canvas = tk.Canvas(win_prob_frame, width=150, height=20, bg="white", 
                           highlightthickness=1, highlightbackground="#999")
win_prob_canvas.pack(side=tk.TOP, anchor=tk.W, pady=2)
win_prob_canvas.create_rectangle(0, 0, 75, 20, fill="green", outline="")
win_prob_canvas.create_rectangle(75, 0, 150, 20, fill="white", outline="")

# # Stockfish skill level display
# skill_level_label = tk.Label(stats_frame, text=f"Bot Level: {STOCKFISH_SKILL_LEVEL}", 
#                             font=("Helvetica", 12), bg="#f0f0f0")
# skill_level_label.pack(side=tk.TOP, anchor=tk.W, pady=2)

# Move classifications section
classification_frame = tk.Frame(metrics_frame, bg="#f0f0f0", relief=tk.GROOVE, bd=2)
classification_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

# Title for move classifications
move_class_title = tk.Label(classification_frame, text="Move Classifications", 
                           font=("Helvetica", 12, "bold"), bg="#f0f0f0")
move_class_title.pack(side=tk.TOP, pady=2)

# Create frames for each classification type with indicators
advantage_frame = tk.Frame(classification_frame, bg="#f0f0f0")
advantage_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
advantage_label = tk.Label(advantage_frame, text="Advantage: 0", font=("Helvetica", 11), 
                          fg="#007700", bg="#f0f0f0", width=18, anchor="w")
advantage_label.pack(side=tk.LEFT)
advantage_indicator = tk.Label(advantage_frame, text="", font=("Helvetica", 11, "bold"), 
                              bg="#f0f0f0")
advantage_indicator.pack(side=tk.LEFT)

slight_advantage_frame = tk.Frame(classification_frame, bg="#f0f0f0")
slight_advantage_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
slight_advantage_label = tk.Label(slight_advantage_frame, text="Slight Advantage: 0", font=("Helvetica", 11), 
                                 fg="#0000FF", bg="#f0f0f0", width=18, anchor="w")
slight_advantage_label.pack(side=tk.LEFT)
slight_advantage_indicator = tk.Label(slight_advantage_frame, text="", font=("Helvetica", 11, "bold"), 
                                     bg="#f0f0f0")
slight_advantage_indicator.pack(side=tk.LEFT)

neutral_frame = tk.Frame(classification_frame, bg="#f0f0f0")
neutral_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
neutral_label = tk.Label(neutral_frame, text="Neutral: 0", font=("Helvetica", 11), 
                        fg="#00AAFF", bg="#f0f0f0", width=18, anchor="w")
neutral_label.pack(side=tk.LEFT)
neutral_indicator = tk.Label(neutral_frame, text="", font=("Helvetica", 11, "bold"), 
                            bg="#f0f0f0")
neutral_indicator.pack(side=tk.LEFT)

slight_disadvantage_frame = tk.Frame(classification_frame, bg="#f0f0f0")
slight_disadvantage_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
slight_disadvantage_label = tk.Label(slight_disadvantage_frame, text="Slight Disadvantage: 0", font=("Helvetica", 11), 
                                    fg="#FF8800", bg="#f0f0f0", width=18, anchor="w")
slight_disadvantage_label.pack(side=tk.LEFT)
slight_disadvantage_indicator = tk.Label(slight_disadvantage_frame, text="", font=("Helvetica", 11, "bold"), 
                                        bg="#f0f0f0")
slight_disadvantage_indicator.pack(side=tk.LEFT)

disadvantage_frame = tk.Frame(classification_frame, bg="#f0f0f0")
disadvantage_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
disadvantage_label = tk.Label(disadvantage_frame, text="Disadvantage: 0", font=("Helvetica", 11), 
                             fg="#FF0000", bg="#f0f0f0", width=18, anchor="w")
disadvantage_label.pack(side=tk.LEFT)
disadvantage_indicator = tk.Label(disadvantage_frame, text="", font=("Helvetica", 11, "bold"), 
                                 bg="#f0f0f0")
disadvantage_indicator.pack(side=tk.LEFT)

# Create a frame for buttons
button_frame = tk.Frame(main_frame, bg="#f0f0f0")
button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

# Add restart button
restart_button = tk.Button(button_frame, text="Restart", font=("Helvetica", 10), command=restart_game)
restart_button.pack(side=tk.LEFT, padx=5)

# Create help button
help_button = tk.Button(button_frame, text="Help", font=("Helvetica", 10), command=show_help)
help_button.pack(side=tk.RIGHT, padx=5)

# Bind the click event to the canvas
canvas.bind("<Button-1>", on_square_click)

# Draw the initial board
draw_board(canvas, board)

# Initialize score bar with center line
center = 100  # Half of bar_height (200)
score_canvas.create_rectangle(0, 0, 50, center, fill="black", outline="")
score_canvas.create_rectangle(0, center, 50, 200, fill="white", outline="")
score_canvas.create_line(0, center, 50, center, fill="gray")

# Initialize win probability bar
win_prob_canvas.create_rectangle(0, 0, 75, 20, fill="green", outline="")
win_prob_canvas.create_rectangle(75, 0, 150, 20, fill="white", outline="")

# Start the timer update
time_update_id = root.after(1000, update_time_elapsed)

# Run the application
root.mainloop()