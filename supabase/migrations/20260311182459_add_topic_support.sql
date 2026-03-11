/*
  # Add Multi-Topic Support to Neural Network Learn Platform

  1. Changes to Existing Tables
    - Add `topic` column to user_progress tables to track progress per topic
    - Update completed_lessons to include topic information

  2. New Columns
    - `current_topic` (text) - Track which topic the user is currently learning
    - Modified lesson/badge/challenge IDs to include topic prefix (e.g., 'cnn-lesson-1', 'ann-lesson-1')

  3. Security
    - Maintain existing RLS policies
    - No changes to security model

  4. Notes
    - Existing CNN data remains compatible
    - New topics: ann, cnn, rnn, vae, transformers
*/

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'user_progress' AND column_name = 'current_topic'
  ) THEN
    ALTER TABLE user_progress ADD COLUMN current_topic text DEFAULT 'cnn';
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_user_progress_current_topic ON user_progress(current_topic);
