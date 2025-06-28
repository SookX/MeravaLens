import { Button, Stack, TextField } from "@mui/material"

const AccountForm = ({ inputs = [], handleSubmit = () => {} }) => {
    return (
        <Stack>
            {
                inputs.map((input, i) => (
                    <TextField
                        key={i}
                        type={input.type}
                        variant="outlined"
                        label={input.label}
                        value={input.value}
                        onChange={(e) => input.setValue(e.target.value)}
                    />
                ))
            }
            <Button variant="contained" onClick={handleSubmit}>Submit</Button>
        </Stack>
    )
}

export default AccountForm