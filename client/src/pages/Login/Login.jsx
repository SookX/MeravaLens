import { useContext, useState } from "react"
import AccountForm from "../../components/AccountForm/AccountForm"
import { DataContext } from "../../context/DataContext"

const Login = () => {
    // Gets global data from the context
    const { crud } = useContext(DataContext)

    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")

    const inputs = [
        {
            type: "email",
            label: "Email",
            value: email,
            setValue: setEmail
        },
        {
            type: "password",
            label: "Password",
            value: password,
            setValue: setPassword
        }
    ]

    const handleSubmit = async () => {
        const response = await crud({
            url: "/login",
            method: "post",
            body: {
                email,
                password,
            }
        })

        console.log(response)
    }

    return (
        <AccountForm inputs={inputs} handleSubmit={handleSubmit} />
    )
}

export default Login