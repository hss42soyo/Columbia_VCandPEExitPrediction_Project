% (c) 2027, Michael Robbins
%% Leakage-Controlled Prediction in Private Markets
% The code directory retains the original `VC & PE Exit Prediction` folder name for backward compatibility with existing repository paths; the manuscript title is now `Leakage-Controlled Prediction in Private Markets`.
% Chapter 9 MATLAB companion wrapper over the canonical Python implementation.

data_mode = "sample";  % "sample" or "live"
live_data_dir = "";    % optional override for live mode

this_dir = fileparts(mfilename("fullpath"));
chapter_root = fileparts(this_dir);
python_dir = fullfile(chapter_root, "python");
runner_path = fullfile(python_dir, "run_vc_pe_exit_prediction.py");
if data_mode == "sample"
    output_dir = fullfile(this_dir, "rendered-sample");
else
    output_dir = fullfile(this_dir, "rendered-live");
end
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

parts = [
    "python",
    '"' + string(runner_path) + '"',
    "--data-mode",
    data_mode,
    "--outdir",
    '"' + string(output_dir) + '"'
];
if strlength(live_data_dir) > 0
    parts = [
        parts
        "--live-data-dir"
        '"' + string(live_data_dir) + '"'
    ];
end
command = strjoin(parts, " ");
status = system(command);
if status ~= 0
    error("Chapter 9 Python companion run failed with exit code %d.", status);
end

disp("Chapter 9 figure package written to:");
disp(output_dir);
